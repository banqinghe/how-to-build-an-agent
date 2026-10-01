import * as readline from 'node:readline/promises';
import * as fs from 'node:fs/promises'
import * as z from 'zod';

const BASE_URL = 'https://api.deepseek.com/chat/completions';
const MODEL = 'deepseek-flash';

type SystemMessage = { role: 'system'; content: string };
type UserMessage = { role: 'user'; content: string };
type AssistantMessage = {
    role: 'assistant';
    content: string | null;
    tool_calls?: ToolCall[];
    reasoning_content?: string | null;
};
type ToolMessage = {
    role: 'tool';
    content: string | null;
    tool_call_id: string
};
type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

type ToolCall = {
    id: string;
    type: 'function';
    function: {
        name: string;
        // JSON string
        // the model does not always generate valid JSON, and may hallucinate parameters not defined
        // by your function schema. Validate the arguments in your code before calling your function.
        arguments: string;
    }
}
type ToolDefinition = {
    name: string;
    description?: string;
    inputSchema: z.ZodType;
    execute: (input: unknown) => Promise<string>
}
type Tool = {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters: Record<string, unknown>; // json schema
    }
}

// ToolDefinition -> Tool
function toWireTool(def: ToolDefinition): Tool {
    return {
        type: 'function',
        function: {
            name: def.name,
            description: def.description,
            parameters: z.toJSONSchema(def.inputSchema),
        }
    }
}

// ====== Read File Tool ======
const readFileInputSchema = z.object({
    path: z.string().describe('The relative path of a file in the working directory.'),
});
const readFileDefinition: ToolDefinition = {
    name: 'read_file',
    description: 'Read the contents of a given relative file path. Use this when you want to see what\'s inside a file. Do not use this with directory names.',
    inputSchema: readFileInputSchema,
    execute: readFile,
};
async function readFile(input: unknown): Promise<string> {
    const { path } = readFileInputSchema.parse(input);
    const content = await fs.readFile(path, 'utf-8');
    return content;
}

// ====== List File Tool ======
const listFileInputSchema = z.object({
    path: z.string().describe('The relative path of a directory in the working directory.'),
});
const listFileDefinition: ToolDefinition = {
    name: 'list_file',
    description: 'List the contents of a given relative directory path. Use this when you want to see what files are inside a directory. Do not use this with file paths.',
    inputSchema: listFileInputSchema,
    execute: listFile,
};
async function listFile(input: unknown): Promise<string> {
    const { path } = listFileInputSchema.parse(input);
    const content = await fs.readdir(path, { withFileTypes: true });
    return JSON.stringify(content, null, 2);
}
// ====== Edit File Tool ======
const editFileInputSchema = z.object({
    path: z.string().describe('The relative path of a file in the working directory.'),
    oldString: z.string().describe('Text to search for - must match exactly and must only have one match exactly'),
    newString: z.string().describe('Text to replace the oldString with.'),
});
const editFileDefinition: ToolDefinition = {
    name: 'edit_file',
    description: `Make edits to a text file.

Replaces 'oldString' with 'newString' in the given file. 'oldString' and 'newString' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.`,
    inputSchema: editFileInputSchema,
    execute: editFile,
};
async function editFile(input: unknown): Promise<string> {
    const { path, oldString, newString } = editFileInputSchema.parse(input);
    let content = '';
    try {
        content = await fs.readFile(path, 'utf-8');
    } catch (err) {
        // If the file doesn't exist, it will be created when writing.
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
            content = '';
        } else {
            throw new Error(`Failed to read the file at path "${path}": ${(err as Error).message}`);
        }
    }
    if (content.includes(oldString)) {
        content = content.replace(oldString, newString);
        await fs.writeFile(path, content, 'utf-8');
    } else {
        throw new Error(`The string "${oldString}" was not found in the file.`);
    }
    return content;
}

type ChatCompletion = {
    choices: Array<{
        message: AssistantMessage;
        finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call';
        index: number;
    }>;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
};

class Agent {
    getUserMessage: () => Promise<string | null>;
    tools: ToolDefinition[] = [];

    constructor(
        getUserMessage: () => Promise<string | null>,
        tools: ToolDefinition[] = []
    ) {
        this.getUserMessage = getUserMessage;
        this.tools = tools;
    }

    async run(_context: any) {
        const conversation: ChatMessage[] = [];

        console.log('Chat with Deepseek (use \'ctrl-d\' to quit)\n');

        let readUserInput = true; // flag

        while (true) {
            if (readUserInput) {
                const userInput = await this.getUserMessage();
                if (!userInput) {
                    fs.appendFile('conversation.txt', '========== END =========\n');
                    break;
                }
                console.log();
                const userMessage: ChatMessage = { role: 'user', content: userInput };
                fs.appendFile('conversation.txt', JSON.stringify(userMessage, null, 2) + '\n');
                conversation.push(userMessage);
            }

            const message = await this.runInference(_context, conversation);
            conversation.push(message);
            fs.appendFile('conversation.txt', JSON.stringify(message, null, 2) + '\n');

            if (message.reasoning_content) {
                // gray text for reasoning content
                console.log(`\x1b[90m${message.reasoning_content}\x1b[0m\n`);
            }

            if (message.content) {
                console.log(`\x1b[93mDeepseek\x1b[0m: ${message.content ?? ''}\n`);
            }
            
            const calls = message.tool_calls ?? [];
            if (calls.length === 0) {
                readUserInput = true;
                continue; // no tool call, read user input
            }
            for (const call of calls) {
                const tool = this.tools.find(t => t.name === call.function.name)!;
                let result: string;
                try {
                    const args = JSON.parse(call.function.arguments);
                    console.log(`\u001b[92mtool\u001b[0m: ${call.function.name}(${call.function.arguments})\n`);
                    result = await tool.execute(args);
                } catch (err) {
                    result = `Error executing tool ${tool.name}: ${err}`;
                }
                const toolMessage: ToolMessage = {
                    role: 'tool',
                    content: result,
                    tool_call_id: call.id,
                };
                conversation.push(toolMessage);
                fs.appendFile('conversation.txt', JSON.stringify(toolMessage, null, 2) + '\n');
            }
            readUserInput = false; // if there are tool calls, do not read user input until the tool calls are resolved
        }
    }

    async runInference(_context: any, conversation: ChatMessage[]): Promise<AssistantMessage> {
        const tools = this.tools.map(t => toWireTool(t));
        const response = await fetch(BASE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
            },
            body: JSON.stringify({
                model: MODEL,
                messages: conversation,
                tools,
            }),
        });

        if (!response.ok) {
            throw new Error(`DeepSeek ${response.status}: ${await response.text()}`);
        }

        const data = (await response.json()) as ChatCompletion;

        const message = data.choices[0]?.message;
        if (!message) {
            throw new Error('DeepSeek 返回了空的 choices');
        }

        return message;
    }
}

function createTerminalSource(rl: readline.Interface) {
    return async function(): Promise<string | null> {
        let input = '';
        try {
            input = await rl.question('\u001b[94mYou\u001b[0m: ');
        } catch {
            return null;
        }
        return input.trim();
    };
}

function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    const tools: ToolDefinition[] = [
        readFileDefinition,
        listFileDefinition,
        editFileDefinition,
    ];
    fs.writeFile('conversation.txt', '========== START =========\n');
    fs.appendFile('conversation.txt', 'tools: ' + JSON.stringify(tools, null, 2) + '\n');
    const getUserMessage = createTerminalSource(rl);
    const agent = new Agent(getUserMessage, tools);
    agent
        .run({})
        .finally(() => rl.close());
}

main();

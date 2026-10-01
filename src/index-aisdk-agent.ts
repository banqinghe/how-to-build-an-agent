import * as readline from 'node:readline/promises';
import * as fs from 'node:fs/promises'
import * as z from 'zod';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { tool, isLoopFinished, ToolLoopAgent } from 'ai';
import type { ModelMessage } from 'ai';

const deepseek = createDeepSeek();
const model = deepseek('deepseek-flash');

const readFileTool = tool({
    description: 'Read the contents of a given relative file path. Use this when you want to see what\'s inside a file. Do not use this with directory names.',
    inputSchema: z.object({
        path: z.string().describe('The relative path of a file in the working directory.'),
    }),
    execute: async ({ path }) => fs.readFile(path, 'utf-8'),
});

const listFileTool = tool({
    description: 'List the contents of a given relative directory path. Use this when you want to see what files are inside a directory. Do not use this with file paths.',
    inputSchema: z.object({
        path: z.string().describe('The relative path of a directory in the working directory.'),
    }),
    execute: async ({ path }) => {
        const content = await fs.readdir(path, { withFileTypes: true });
        return JSON.stringify(content, null, 2);
    }
});

const editFileTool = tool({
    description: `Make edits to a text file.

Replaces 'oldString' with 'newString' in the given file. 'oldString' and 'newString' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.`,
    inputSchema: z.object({
        path: z.string().describe('The relative path of a file in the working directory.'),
        oldString: z.string().describe('Text to search for - must match exactly and must only have one match exactly'),
        newString: z.string().describe('Text to replace the oldString with.'),
    }),
    execute: async ({ path, oldString, newString }) => {
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
    },
});

const tools = {
    read_file: readFileTool,
    list_file: listFileTool,
    edit_file: editFileTool,
};

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

const agent = new ToolLoopAgent({
    model,
    tools,
    stopWhen: isLoopFinished(),
    onLanguageModelCallEnd: ({ content }) => {
        let reasoningText = '';
        let text = '';
        for (const part of content) {
            if (part.type === 'reasoning') {
                reasoningText = part.text;
            } else if (part.type === 'text') {
                text = part.text;
            }
        }
        if (reasoningText) {
            console.log(`\x1b[90m${reasoningText}\x1b[0m\n`);
        }
        if (text) {
            console.log(`\x1b[93mDeepseek\x1b[0m: ${text}\n`);
        }
    },
    // call `onToolExecutionStart` before tool calls, it's the time when we print tool call info
    onToolExecutionStart: ({ toolCall }) => {
        console.log(`\u001b[92mtool\u001b[0m: ${toolCall.toolName}(${JSON.stringify(toolCall.input)})\n`);
    },
});

async function chat(getUserMessage: () => Promise<string | null>) {
    console.log('Chat with Deepseek (use \'ctrl-d\' to quit)\n');

    const conversation: ModelMessage[] = [];
    while (true) {
        const userInput = await getUserMessage();
        if (!userInput) {
            break;
        }

        const userMessage: ModelMessage = {
            role: 'user',
            content: userInput,
        };
        conversation.push(userMessage);

        const result = await agent.generate({ messages: conversation });
        conversation.push(...result.responseMessages);
    }
}

function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    const getUserMessage = createTerminalSource(rl);
    chat(getUserMessage)
        .finally(() => rl.close());
}

main();

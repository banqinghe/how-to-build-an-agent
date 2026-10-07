import * as readline from 'node:readline/promises';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { createModels, Type, validateToolArguments } from '@earendil-works/pi-ai';
import { deepseekProvider } from '@earendil-works/pi-ai/providers/deepseek';
import type { Message, ToolCall, UserMessage, Static, Tool, AssistantMessage, TSchema, ToolResultMessage } from '@earendil-works/pi-ai';

type GetUserMessageFunction = () => Promise<string | null>;

type ExecutableTool<S extends TSchema = TSchema> = Tool<S> & {
    execute(args: Static<S>): Promise<string>;
}

function defineTool<S extends TSchema>(tool: ExecutableTool<S>): ExecutableTool<S> {
    return tool;
}

type BashError = {
    stdout?: string;
    stderr?: string;
    code?: number;
    killed?: boolean;
};

const execAsync = promisify(exec);
const MAX_OUTPUT = 30_000;

function tail(text: string) {
    return text.length > MAX_OUTPUT
        ? `[truncated, showing last ${MAX_OUTPUT} chars]\n${text.slice(-MAX_OUTPUT)}`
        : text;
}

const bashTool = defineTool({
    name: 'bash',
    description: 'Execute a bash command in the current working directory and return its stdout and stderr. Do not run interactive commands or commands that never exit (dev servers, watchers).',
    parameters: Type.Object({
        command: Type.String({ description: 'The bash command to execute.' }),
        timeout: Type.Optional(
            Type.Number({ description: 'Timeout in seconds. Defaults to 30.' })
        ),
    }),
    execute: async ({ command, timeout = 30 }) => {
        try {
            const { stdout, stderr } = await execAsync(
                command,
                {
                    shell: '/bin/bash',
                    timeout: timeout * 1000,
                    maxBuffer: 10 * 1024 * 1024,
                },
            );
            return tail(stdout + stderr) || '(no output)';
        } catch (err) {
            const { stdout = '', stderr = '', code, killed } = err as BashError;
            const status = killed ? `Command timed out after ${timeout} seconds` : `Exit code ${code}`;
            throw new Error(`${tail(stdout + stderr)}\n\n${status}`);
        }
    },
});

const tools: ExecutableTool[] = [
    bashTool,
];

class Agent {
    getUserMessage: GetUserMessageFunction;
    chat: (messages: Message[]) => Promise<AssistantMessage>;

    constructor(getUserMessage: GetUserMessageFunction) {
        const models = createModels();
        models.setProvider(deepseekProvider());
        const model = models.getModel('deepseek', 'deepseek-flash');

        if (!model) {
            throw new Error('Unknow model: deepseek/deepseek-flash');
        }

        this.chat = messages => models.completeSimple(
            model,
            {
                systemPrompt: 'Your are a helpful assistant',
                messages,
                tools,
            },
        );

        this.getUserMessage = getUserMessage;
    }

    async run() {
        const messages: Message[] = [];

        console.log('Chat with Deepseek (use \'ctrl-d\' to quit)\n');

        let hasToolExecution = false;

        while (true) {
            if (!hasToolExecution) {
                const userInput = await this.getUserMessage();
                if (!userInput) {
                    break;
                }
                const userMessage: UserMessage = {
                    role: 'user',
                    content: userInput,
                    timestamp: Date.now(),
                };
                messages.push(userMessage);
            }

            // call llm endpoint
            // append:
            //    a. user input
            // or b. tool result
            const result = await this.chat(messages);
            messages.push(result);

            if (result.stopReason === 'error') {
                // 红色输出？
                console.error(`\x1b[91merror ${result.errorMessage}\x1b[0m`);
                hasToolExecution = false;
                continue;
            }

            const calls: ToolCall[] = [];

            for (const content of result.content) {
                switch (content.type) {
                    case 'thinking':
                        console.log(`\x1b[90m${content.thinking}\x1b[0m\n`);
                        break;
                    case 'text':
                        console.log(`\x1b[93mDeepseek\x1b[0m: ${content.text}\n`);
                        break;
                    case 'toolCall':
                        calls.push(content);
                }
            }

            for (const call of calls) {
                const tool = tools.find(t => t.name === call.name);
                let toolResult = '';
                let isError = false;
                if (!tool) {
                    toolResult = `${call.name} is a non-existent tool`;
                    isError = true;
                } else {
                    try {
                        console.log(`\u001b[92mtool\u001b[0m: ${call.name}(${JSON.stringify(call.arguments)})\n`);
                        toolResult = await tool.execute(validateToolArguments(tool, call));
                    } catch (err) {
                        toolResult = `Error executing tool ${tool.name}: ${err}`;
                        isError = true;
                    }
                }
                const toolResultMessage: ToolResultMessage = {
                    role: 'toolResult',
                    toolCallId: call.id,
                    toolName: call.name,
                    content: [{ type: 'text', text: toolResult }],
                    isError,
                    timestamp: Date.now(),
                };
                messages.push(toolResultMessage);
            }

            hasToolExecution = calls.length > 0;
        }
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
    const getUserMessage = createTerminalSource(rl);

    const agent = new Agent(getUserMessage);
    agent
        .run()
        .finally(() => rl.close());
}

main();

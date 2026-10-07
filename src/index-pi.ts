import * as readline from 'node:readline/promises';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { createModels, Type, validateToolArguments } from '@earendil-works/pi-ai';
import { deepseekProvider } from '@earendil-works/pi-ai/providers/deepseek';
import type {
    Api,
    Context,
    Model,
    MutableModels,
    Static,
    TSchema,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
} from '@earendil-works/pi-ai';

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

const defaultTools: ExecutableTool[] = [
    bashTool,
];

const SYSTEM_PROMPT = 'You are a helpful assistant';

class Agent {
    models: MutableModels;
    model: Model<Api>;
    context: Context;

    constructor(tools: ExecutableTool[] = defaultTools, systemPrompt: string = SYSTEM_PROMPT) {
        this.models = createModels();
        this.models.setProvider(deepseekProvider());

        const model = this.models.getModel('deepseek', 'deepseek-flash');
        if (!model) {
            throw new Error('Unknown model: deepseek/deepseek-flash');
        }
        this.model = model;

        // Single source of truth: the same list is sent to the model and used to look up tools to execute.
        this.context = {
            systemPrompt,
            messages: [],
            tools,
        };
    }

    /** Start a new session: clear the conversation history, keep systemPrompt and tools. */
    reset() {
        this.context = { ...this.context, messages: [] };
    }

    /** Push a user message, then keep running until the model stops calling tools. */
    async prompt(input: string) {
        const userMessage: UserMessage = {
            role: 'user',
            content: input,
            timestamp: Date.now(),
        };
        this.context.messages.push(userMessage);
        await this.runAgentLoop();
    }

    async runAgentLoop() {
        while (true) {
            const result = await this.models.completeSimple(this.model, this.context);
            this.context.messages.push(result);

            if (result.stopReason === 'error') {
                console.error(`\x1b[91merror ${result.errorMessage}\x1b[0m`);
                return;
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
                this.context.messages.push(await this.executeTool(call));
            }

            // No more tool calls -> this turn is done; go back to main and wait for the next input.
            if (calls.length === 0) {
                return;
            }
        }
    }

    async executeTool(call: ToolCall): Promise<ToolResultMessage> {
        const tool = this.context.tools?.find(t => t.name === call.name) as ExecutableTool | undefined;

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

        return {
            role: 'toolResult',
            toolCallId: call.id,
            toolName: call.name,
            content: [{ type: 'text', text: toolResult }],
            isError,
            timestamp: Date.now(),
        };
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

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    const getUserMessage = createTerminalSource(rl);

    const agent = new Agent();

    console.log('Chat with Deepseek (use \'ctrl-d\' to quit)\n');

    while (true) {
        const userInput = await getUserMessage();
        if (!userInput) {
            break;
        }
        await agent.prompt(userInput);
    }

    rl.close();
}

main();

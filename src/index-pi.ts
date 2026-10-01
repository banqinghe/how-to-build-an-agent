import * as readline from 'node:readline/promises';
import * as fs from 'node:fs/promises';
import { createModels, Type } from '@earendil-works/pi-ai';
import { deepseekProvider } from '@earendil-works/pi-ai/providers/deepseek';
import { Agent, type AgentTool } from '@earendil-works/pi-agent-core';

// ====== Provider ======
// Models 是一个 provider 集合：provider 自己带着模型清单、认证方式和线协议。
// DeepSeek 的认证走 DEEPSEEK_API_KEY，这里不需要手动传 key。
const models = createModels();
models.setProvider(deepseekProvider());
const model = models.getModel('deepseek', 'deepseek-flash');
if (!model) {
    throw new Error('Unknown model: deepseek/deepseek-flash');
}

// ====== Read File Tool ======
const readFileInputSchema = Type.Object({
    path: Type.String({ description: 'The relative path of a file in the working directory.' }),
});
const readFileTool: AgentTool<typeof readFileInputSchema> = {
    name: 'read_file',
    label: 'Read file',
    description: 'Read the contents of a given relative file path. Use this when you want to see what\'s inside a file. Do not use this with directory names.',
    parameters: readFileInputSchema,
    execute: async (_toolCallId, { path }) => ({
        content: [{ type: 'text', text: await fs.readFile(path, 'utf-8') }],
        details: { path },
    }),
};

// ====== List File Tool ======
const listFileInputSchema = Type.Object({
    path: Type.String({ description: 'The relative path of a directory in the working directory.' }),
});
const listFileTool: AgentTool<typeof listFileInputSchema> = {
    name: 'list_file',
    label: 'List directory',
    description: 'List the contents of a given relative directory path. Use this when you want to see what files are inside a directory. Do not use this with file paths.',
    parameters: listFileInputSchema,
    execute: async (_toolCallId, { path }) => {
        const content = await fs.readdir(path, { withFileTypes: true });
        return {
            content: [{ type: 'text', text: JSON.stringify(content, null, 2) }],
            details: { path },
        };
    },
};

// ====== Edit File Tool ======
const editFileInputSchema = Type.Object({
    path: Type.String({ description: 'The relative path of a file in the working directory.' }),
    oldString: Type.String({ description: 'Text to search for - must match exactly and must only have one match exactly' }),
    newString: Type.String({ description: 'Text to replace the oldString with.' }),
});
const editFileTool: AgentTool<typeof editFileInputSchema> = {
    name: 'edit_file',
    label: 'Edit file',
    description: `Make edits to a text file.

Replaces 'oldString' with 'newString' in the given file. 'oldString' and 'newString' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.`,
    parameters: editFileInputSchema,
    execute: async (_toolCallId, { path, oldString, newString }) => {
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
        if (!content.includes(oldString)) {
            throw new Error(`The string "${oldString}" was not found in the file.`);
        }
        const nextContent = content.replace(oldString, newString);
        await fs.writeFile(path, nextContent, 'utf-8');
        return {
            content: [{ type: 'text', text: nextContent }],
            details: { path },
        };
    },
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

// ====== Agent ======
// agent 自己持有整个 transcript（agent.state.messages），
// 并且 prompt() 会一直跑到模型给出最终答复，tool loop 在它内部。
const agent = new Agent({
    initialState: {
        model,
        tools: [readFileTool, listFileTool, editFileTool],
        thinkingLevel: 'low',
    },
    streamFn: models.streamSimple.bind(models),
});

const RESET = '\x1b[0m';
let reasoningOpen = false;

// 两条注意事项：
// 1. thinking_start 会先于内容发出，有时候那个思考块里一个字都没有
//    （实测 turn 1 就是 thinking_start 之后直接 text_start），
//    所以灰色转义码要等到第一个 thinking_delta 才写，否则会多出空行。
// 2. thinking_end 的时机不可靠，它可能晚于 text_start 才发，
//    所以只要开始写别的东西，就把灰色这一段收掉。
function closeReasoning() {
    if (!reasoningOpen) {
        return;
    }
    process.stdout.write(`${RESET}\n\n`);
    reasoningOpen = false;
}

agent.subscribe((event) => {
    if (event.type === 'message_update') {
        const part = event.assistantMessageEvent;
        switch (part.type) {
            case 'thinking_delta':
                if (!reasoningOpen) {
                    process.stdout.write('\x1b[90m');
                    reasoningOpen = true;
                }
                process.stdout.write(part.delta);
                break;
            case 'thinking_end':
                closeReasoning();
                break;
            case 'text_start':
                closeReasoning();
                process.stdout.write(`\x1b[93mDeepseek${RESET}: `);
                break;
            case 'text_delta':
                process.stdout.write(part.delta);
                break;
            case 'text_end':
                process.stdout.write('\n\n');
                break;
            default:
                break;
        }
    }

    // tool calls：在工具真正执行前打印，和 index-aisdk-agent 的 onToolExecutionStart 对齐
    if (event.type === 'tool_execution_start') {
        closeReasoning();
        console.log(`\u001b[92mtool${RESET}: ${event.toolName}(${JSON.stringify(event.args)})\n`);
    }

    // 兜底：万一某个 provider 没发 thinking_end / text_start，别让灰色一直挂着
    if (event.type === 'message_end') {
        closeReasoning();
    }

    // pi 不像 Vercel 那样抛异常，失败会编码成一条 assistant 消息，这里得自己看
    if (event.type === 'message_end' && event.message.role === 'assistant') {
        if (event.message.stopReason === 'error') {
            console.error(`\x1b[91merror\x1b[0m: ${event.message.errorMessage}`);
        }
    }
});

async function chat(getUserMessage: () => Promise<string | null>) {
    console.log('Chat with Deepseek (use \'ctrl-d\' to quit)\n');

    while (true) {
        const userInput = await getUserMessage();
        if (!userInput) {
            break;
        }

        // 不用再维护 conversation，也不用再手写 tool loop
        await agent.prompt(userInput);
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

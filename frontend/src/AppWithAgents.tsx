import {useEffect, useRef, useState} from "react";
import {Card} from "@/components/ui/card";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Input} from "@/components/ui/input";
import {Button} from "@/components/ui/button";
import {Avatar, AvatarFallback} from "@/components/ui/avatar";
import {Response} from "@/components/ui/shadcn-io/ai/response";
import {Badge} from "@/components/ui/badge";
import {
    InlineCitation,
    InlineCitationCard,
    InlineCitationCardBody,
    InlineCitationCardTrigger,
    InlineCitationCarousel,
    InlineCitationCarouselContent,
    InlineCitationCarouselHeader,
    InlineCitationCarouselIndex,
    InlineCitationCarouselItem,
    InlineCitationCarouselNext,
    InlineCitationCarouselPrev,
    InlineCitationSource,
    InlineCitationText,
} from '@/components/ui/shadcn-io/ai/inline-citation';
import {
    Task,
    TaskTrigger,
    TaskContent,
    TaskItem,
    TaskItemFile,
} from '@/components/ui/shadcn-io/ai/task';

type Source = { id: number | string; content: string; metadata?: Record<string, any> };
type Task = { id: string; title: string; description: string; status: "in_progress" | "completed" | "error" };
type Msg = {
    role: "user" | "assistant";
    content: string;
    sources?: Source[];
    metadata?: any;
    showTask?: boolean;
    taskOnly?: boolean;
    tasks?: Task[];  // NEW: Dynamic tasks
};
type ApiResponse = { result?: string; sources?: Source[] } | string[] | string | unknown;

type Mode = "chat" | "classify" | "full-flow";

function toText(data: ApiResponse): string {
    if (typeof data === "string") return data;
    if (Array.isArray(data)) return data.map(String).join("\n");
    if (data && typeof data === "object" && "result" in data && typeof (data as any).result === "string")
        return (data as any).result;
    return JSON.stringify(data);
}

function toSources(data: ApiResponse): Source[] | undefined {
    if (data && typeof data === "object" && "sources" in data && Array.isArray((data as any).sources)) {
        return (data as any).sources;
    }
    return undefined;
}

export default function App() {
    const [mode, setMode] = useState<Mode>("chat");
    const [messages, setMessages] = useState<Msg[]>([
        {role: "assistant", content: "Hi! Ask me anything about the EU AI Act."},
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [classificationResult, setClassificationResult] = useState<any>(null);

    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({behavior: "smooth"});
    }, [messages, loading]);

    // Reset when changing modes
    useEffect(() => {
        if (mode === "chat") {
            setMessages([{role: "assistant", content: "Hi! Ask me anything about the EU AI Act."}]);
        } else if (mode === "classify") {
            setMessages([{
                role: "assistant",
                content: "I'll help classify your AI system. Please describe your AI system in detail."
            }]);
        } else if (mode === "full-flow") {
            setMessages([{
                role: "assistant",
                content: "I'll classify your AI system and generate a compliance checklist. Please describe your AI system."
            }]);
        }
        setInput("");
        setClassificationResult(null);
    }, [mode]);

    async function send() {
        const text = input.trim();
        if (!text || loading) return;

        const next: Msg[] = [...messages, {role: "user" as const, content: text}];
        setMessages(next);
        setInput("");
        setLoading(true);

        try {
            if (mode === "chat") {
                await sendChat(text, next);
            } else if (mode === "classify") {
                await sendClassify(text);
            } else if (mode === "full-flow") {
                await sendFullFlow(text);
            }
        } catch (e: any) {
            setMessages(m => [...m, {role: "assistant", content: `Error: ${e?.message ?? e}`}]);
        } finally {
            setLoading(false);
        }
    }

    async function sendChat(text: string, history: Msg[]) {
        const res = await fetch("/api/search/", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({query: text, history: history.map(m => m.content)}),
        });
        const data: ApiResponse = await res.json();
        if (!res.ok) throw new Error(toText(data));
        const answer = toText(data);
        const sources = toSources(data);
        setMessages(m => [...m,
            {role: "assistant", content: "", taskOnly: true},
            {role: "assistant", content: answer, sources}
        ]);
    }

    async function sendClassify(description: string) {
        // Add initial loading message with empty tasks - capture the ACTUAL index
        let taskMessageIndex = -1;
        setMessages(m => {
            taskMessageIndex = m.length;  // Capture correct index from state
            return [...m, {role: "assistant", content: "", taskOnly: true, tasks: []}];
        });

        let taskIdCounter = 0;
        let responseMessageIndex = -1;
        let streamingText = "";

        const res = await fetch("/api/classify/stream", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ai_system_description: description}),
        });

        if (!res.ok) throw new Error("Failed to start classification stream");
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalData: any = null;

        try {
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, {stream: true});
                const lines = buffer.split('\n');
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim() || !line.startsWith('data: ')) continue;

                    const data = JSON.parse(line.substring(6));

                    if (data.type === "task") {
                        // Add or update task using functional state update
                        setMessages(m => {
                            const newMessages = [...m];
                            const currentTasks = newMessages[taskMessageIndex].tasks || [];

                            // Find existing task
                            const existingIndex = currentTasks.findIndex((t: Task) => t.title === data.title);

                            let updatedTasks: Task[];
                            if (existingIndex >= 0) {
                                // Update existing task
                                updatedTasks = [...currentTasks];
                                updatedTasks[existingIndex] = {
                                    ...updatedTasks[existingIndex],
                                    status: data.status,
                                    description: data.description
                                };
                            } else {
                                // Add new task
                                updatedTasks = [...currentTasks, {
                                    id: `task-${taskIdCounter++}`,
                                    title: data.title,
                                    description: data.description,
                                    status: data.status
                                }];
                            }

                            // Update the message with new tasks
                            newMessages[taskMessageIndex] = {
                                ...newMessages[taskMessageIndex],
                                tasks: updatedTasks
                            };

                            return newMessages;
                        });

                    } else if (data.type === "token") {
                        // Stream LLM tokens
                        streamingText += data.token;

                        // Create or update streaming response message
                        if (responseMessageIndex === -1) {
                            setMessages(m => {
                                responseMessageIndex = m.length;
                                return [...m, {role: "assistant", content: streamingText}];
                            });
                        } else {
                            setMessages(m => {
                                const newMessages = [...m];
                                newMessages[responseMessageIndex] = {
                                    ...newMessages[responseMessageIndex],
                                    content: streamingText
                                };
                                return newMessages;
                            });
                        }

                    } else if (data.type === "final_result") {
                        finalData = data.data;
                    } else if (data.type === "error") {
                        throw new Error(data.message);
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }

        // Process final result - replace streaming text with formatted response
        if (finalData) {
            setClassificationResult(finalData);

            let response = "";
            if (finalData.needs_more_info) {
                response = `I need more information to classify your system.\n\n**Questions:**\n${finalData.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}`;
            } else {
                response = `**Classification Result**\n\n`;
                response += `**Risk Level:** ${finalData.risk_level}\n`;
                response += `**System Type:** ${finalData.system_type}\n`;
                response += `**Confidence:** ${(finalData.confidence * 100).toFixed(0)}%\n\n`;
                response += `**Reasoning:** ${finalData.reasoning}\n\n`;
                if (finalData.relevant_articles && finalData.relevant_articles.length > 0) {
                    response += `**Relevant Articles:** ${finalData.relevant_articles.join(", ")}`;
                }
            }

            // Replace the streaming message with formatted response
            if (responseMessageIndex !== -1) {
                setMessages(m => {
                    const newMessages = [...m];
                    newMessages[responseMessageIndex] = {
                        ...newMessages[responseMessageIndex],
                        content: response,
                        metadata: finalData
                    };
                    return newMessages;
                });
            } else {
                // No streaming happened, add new message
                setMessages(m => [...m, {role: "assistant", content: response, metadata: finalData}]);
            }
        }
    }

    async function sendFullFlow(description: string) {
        // Add initial loading message with empty tasks - capture the ACTUAL index
        let taskMessageIndex = -1;
        setMessages(m => {
            taskMessageIndex = m.length;  // Capture correct index from state
            return [...m, {role: "assistant", content: "", taskOnly: true, tasks: []}];
        });

        let taskIdCounter = 0;

        // Helper function to add/update task - uses functional state update
        const handleTask = (data: any) => {
            setMessages(m => {
                const newMessages = [...m];

                if (!newMessages[taskMessageIndex]) {
                    console.error("[FULL-FLOW] ERROR: Message at index", taskMessageIndex, "does not exist!");
                    return m;
                }

                const currentTasks = newMessages[taskMessageIndex].tasks || [];

                // Find existing task
                const existingIndex = currentTasks.findIndex((t: Task) => t.title === data.title);

                let updatedTasks: Task[];
                if (existingIndex >= 0) {
                    // Update existing task
                    updatedTasks = [...currentTasks];
                    updatedTasks[existingIndex] = {
                        ...updatedTasks[existingIndex],
                        status: data.status,
                        description: data.description
                    };
                } else {
                    // Add new task
                    updatedTasks = [...currentTasks, {
                        id: `task-${taskIdCounter++}`,
                        title: data.title,
                        description: data.description,
                        status: data.status
                    }];
                }

                // Update the message with new tasks
                newMessages[taskMessageIndex] = {
                    ...newMessages[taskMessageIndex],
                    tasks: updatedTasks
                };

                return newMessages;
            });
        };

        // Helper function to stream from endpoint with streaming text support
        const streamFromEndpoint = async (url: string, body: any, stepTitle: string) => {
            let responseMessageIndex = -1;
            let streamingText = "";

            const res = await fetch(url, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(body),
            });

            if (!res.ok) throw new Error(`Failed to start stream: ${url}`);
            if (!res.body) throw new Error("No response body");

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let finalData: any = null;

            try {
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || "";

                    for (const line of lines) {
                        if (!line.trim() || !line.startsWith('data: ')) continue;
                        const data = JSON.parse(line.substring(6));

                        if (data.type === "task") {
                            handleTask(data);
                        } else if (data.type === "token") {
                            // Stream LLM tokens into a separate message
                            streamingText += data.token;

                            // Create or update streaming response message
                            if (responseMessageIndex === -1) {
                                setMessages(m => {
                                    responseMessageIndex = m.length;
                                    return [...m, {role: "assistant", content: `**${stepTitle}**\n\n${streamingText}`}];
                                });
                            } else {
                                setMessages(m => {
                                    const newMessages = [...m];
                                    newMessages[responseMessageIndex] = {
                                        ...newMessages[responseMessageIndex],
                                        content: `**${stepTitle}**\n\n${streamingText}`
                                    };
                                    return newMessages;
                                });
                            }
                        } else if (data.type === "final_result") {
                            finalData = data.data;
                        } else if (data.type === "error") {
                            throw new Error(data.message);
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }

            return {finalData, responseMessageIndex};
        };

        try {
            // Step 1: Classification (streams in real-time)
            const {finalData: classifyData, responseMessageIndex: classifyMsgIndex} = await streamFromEndpoint(
                "/api/classify/stream",
                {ai_system_description: description},
                "Step 1: Classification Results"
            );

            if (!classifyData) {
                throw new Error("No classification data received");
            }

            if (classifyData.needs_more_info) {
                let response = `I need more information before I can classify your system.\n\n**Questions:**\n${classifyData.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}\n\nPlease provide more details.`;

                // Replace streaming message with formatted response
                if (classifyMsgIndex !== -1) {
                    setMessages(m => {
                        const newMessages = [...m];
                        newMessages[classifyMsgIndex] = {
                            ...newMessages[classifyMsgIndex],
                            content: response
                        };
                        return newMessages;
                    });
                } else {
                    setMessages(m => [...m, {role: "assistant", content: response}]);
                }
                return;
            }

            // Format Step 1 result
            let step1Response = `**Step 1: Classification Results**\n\n`;
            step1Response += `**Risk Level:** ${classifyData.risk_level}\n`;
            step1Response += `**System Type:** ${classifyData.system_type}\n`;
            step1Response += `**Confidence:** ${(classifyData.confidence * 100).toFixed(0)}%\n\n`;
            step1Response += `**Reasoning:** ${classifyData.reasoning}\n\n`;
            if (classifyData.relevant_articles && classifyData.relevant_articles.length > 0) {
                step1Response += `**Relevant Articles:** ${classifyData.relevant_articles.join(", ")}`;
            }

            // Replace streaming message with formatted Step 1
            if (classifyMsgIndex !== -1) {
                setMessages(m => {
                    const newMessages = [...m];
                    newMessages[classifyMsgIndex] = {
                        ...newMessages[classifyMsgIndex],
                        content: step1Response,
                        metadata: classifyData
                    };
                    return newMessages;
                });
            }

            // Step 2: Checklist generation (streams in real-time as separate message)
            const {finalData: checklistData, responseMessageIndex: checklistMsgIndex} = await streamFromEndpoint(
                "/api/checklist/stream",
                {
                    risk_level: classifyData.risk_level,
                    system_type: classifyData.system_type,
                    system_description: description
                },
                "Step 2: Compliance Checklist"
            );

            if (!checklistData) {
                throw new Error("No checklist data received");
            }

            // Format Step 2 result
            let step2Response = `**Step 2: Compliance Checklist**\n\n`;
            step2Response += `**Total Items:** ${checklistData.total_items}\n\n`;
            step2Response += `${checklistData.summary}\n\n`;
            step2Response += `**Requirements:**\n\n`;

            checklistData.checklist_items.forEach((item: any, idx: number) => {
                step2Response += `**${idx + 1}. ${item.requirement}**\n`;
                step2Response += `   - Priority: ${item.priority}\n`;
                step2Response += `   - Category: ${item.category}\n`;
                if (item.applicable_articles && item.applicable_articles.length > 0) {
                    step2Response += `   - Articles: ${item.applicable_articles.join(", ")}\n`;
                }
                step2Response += `\n`;
            });

            // Replace streaming message with formatted Step 2
            if (checklistMsgIndex !== -1) {
                setMessages(m => {
                    const newMessages = [...m];
                    newMessages[checklistMsgIndex] = {
                        ...newMessages[checklistMsgIndex],
                        content: step2Response,
                        metadata: checklistData
                    };
                    return newMessages;
                });
            }

        } catch (error: any) {
            throw error;
        }
    }

    function onKey(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    }

    return (
        <div className="h-screen w-full p-4">
            <div className="mx-auto max-w-3xl h-full min-h-0 space-y-3">
                {/* Mode selector */}
                <div className="flex gap-2 justify-center">
                    <Button
                        variant={mode === "chat" ? "default" : "outline"}
                        onClick={() => setMode("chat")}
                    >
                        Q&A Chat
                    </Button>
                    <Button
                        variant={mode === "classify" ? "default" : "outline"}
                        onClick={() => setMode("classify")}
                    >
                        Classification
                    </Button>
                    <Button
                        variant={mode === "full-flow" ? "default" : "outline"}
                        onClick={() => setMode("full-flow")}
                    >
                        Full Compliance Flow
                    </Button>
                </div>

                <Card className="h-[calc(80vh-3rem)] grid grid-rows-[1fr_auto] overflow-hidden">
                    <div className="min-h-0">
                        <ScrollArea className="h-full">
                            <div className="p-4 space-y-3">
                                {messages.map((m, i) => <Bubble key={i} role={m.role} content={m.content}
                                                                sources={m.sources} showTask={m.showTask} taskOnly={m.taskOnly} mode={mode} tasks={m.tasks}/>)}
                                {loading && mode === "chat" && <LoadingBubble mode={mode}/>}
                                <div ref={endRef}/>
                            </div>
                        </ScrollArea>
                    </div>

                    <div className="border-t p-3 flex gap-2">
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={onKey}
                            placeholder={
                                mode === "chat"
                                    ? "Ask a question about EU AI Act..."
                                    : "Describe your AI system..."
                            }
                        />
                        <Button onClick={send} disabled={loading || !input.trim()}>Send</Button>
                    </div>
                </Card>
            </div>
        </div>
    );

    function LoadingBubble({mode}: { mode: Mode }) {
        // Only used for chat mode (non-streaming)
        return (
            <div className="mb-1 flex items-start gap-3 justify-start">
                <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>
                <div className="max-w-[80%] rounded-2xl px-3 py-2 text-sm bg-muted">
                    <Task>
                        <TaskTrigger title="Processing your request"/>
                        <TaskContent>
                            <TaskItem>Retrieving relevant information</TaskItem>
                        </TaskContent>
                    </Task>
                </div>
            </div>
        );
    }

    function Bubble({
                        role, content, muted, sources, showTask, taskOnly, mode, tasks
                    }: { role: "user" | "assistant"; content: string; muted?: boolean; sources?: Source[]; showTask?: boolean; taskOnly?: boolean; mode?: Mode; tasks?: Task[] }) {
        const isUser = role === "user";

        // If taskOnly is true, only render the Task component
        if (taskOnly && !isUser) {
            return (
                <div className="mb-1 flex items-start gap-3 justify-start">
                    <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>
                    <div className="max-w-[80%] rounded-2xl px-3 py-2 text-sm bg-muted">
                        {/* Render dynamic tasks from streaming */}
                        <Task>
                            <TaskTrigger title={mode === "classify" ? "Classifying AI system" : mode === "full-flow" ? "Running full compliance workflow" : "Processing your request"}/>
                            <TaskContent>
                                {tasks && tasks.length > 0 ? (
                                    tasks.map((task) => (
                                        <TaskItem key={task.id}>
                                            {task.status === "completed" ? "✓ " : task.status === "error" ? "✗ " : "⏳ "}
                                            {task.title}
                                            {task.description && ` - ${task.description}`}
                                        </TaskItem>
                                    ))
                                ) : (
                                    <TaskItem>⏳ Initializing...</TaskItem>
                                )}
                            </TaskContent>
                        </Task>
                    </div>
                </div>
            );
        }

        return (
            <div className={`mb-1 flex items-start gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
                {!isUser && <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>}
                <div
                    className={[
                        "max-w-[80%] rounded-2xl px-3 py-2 text-sm",
                        isUser ? "bg-primary text-primary-foreground" : "bg-muted",
                        muted ? "opacity-60" : "",
                    ].join(" ")}
                >
                    {isUser ? (
                        <span className="whitespace-pre-wrap break-words break-normal">{content}</span>
                    ) : (
                        <>
                            <Response>{content}</Response>
                            {sources && sources.length > 0 && (
                                <InlineCitation>
                                    <InlineCitationText>Sources</InlineCitationText>
                                    <InlineCitationCard>
                                        <InlineCitationCardTrigger sources={[]}/>
                                        <InlineCitationCardBody>
                                            <InlineCitationCarousel>
                                                <InlineCitationCarouselHeader>
                                                    <InlineCitationCarouselPrev/>
                                                    <InlineCitationCarouselNext/>
                                                    <InlineCitationCarouselIndex/>
                                                </InlineCitationCarouselHeader>
                                                <InlineCitationCarouselContent>
                                                    {sources.map((src, idx) => (
                                                        <InlineCitationCarouselItem key={src.id ?? idx}>
                                                            <InlineCitationSource
                                                                title={`Source ${idx + 1} (${src.metadata?.id})`}
                                                                url="#"
                                                                description={src.content}
                                                            />
                                                        </InlineCitationCarouselItem>
                                                    ))}
                                                </InlineCitationCarouselContent>
                                            </InlineCitationCarousel>
                                        </InlineCitationCardBody>
                                    </InlineCitationCard>
                                </InlineCitation>
                            )}
                        </>
                    )}
                </div>
                {isUser && <Avatar className="h-8 w-8"><AvatarFallback>U</AvatarFallback></Avatar>}
            </div>
        );
    }
}

import {useEffect, useRef, useState} from "react";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Avatar, AvatarFallback} from "@/components/ui/avatar";
import {Response} from "@/components/ui/shadcn-io/ai/response";
import {Actions, Action} from "@/components/ui/shadcn-io/ai/actions";
import {Tabs, TabsList, TabsTrigger} from "@/components/ui/tabs";
import {
    PromptInput,
    PromptInputTextarea,
    PromptInputToolbar,
    PromptInputSubmit,
} from "@/components/ui/shadcn-io/ai/prompt-input";
import {Suggestions, Suggestion} from "@/components/ui/shadcn-io/ai/suggestion";
import TypingText from "@/components/ui/shadcn-io/text/typing-text";
import { cn } from "@/lib/utils";
import {CopyIcon} from "lucide-react";
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
} from '@/components/ui/shadcn-io/ai/task';
import { Vortex } from "@/components/ui/vortex";

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

type Mode = "chat" | "compliance-agents";

const CopySuccessIcon = ({className}: {className?: string}) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={cn("lucide lucide-check-icon lucide-check", className)}
    >
        <path d="M20 6 9 17l-5-5"/>
    </svg>
);

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
    const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
    const [inputFocused, setInputFocused] = useState(false);
    const suggestionPresets: Record<Mode, {label: string; prompt: string}[]> = {
        chat: [
            {
                label: "Airport facial ID",
                prompt: "What EU AI Act requirements apply to facial recognition at airports?"
            },
            {
                label: "Biometric obligations",
                prompt: "How does the AI Act treat biometric identification systems?"
            },
            {
                label: "Chatbot transparency",
                prompt: "What transparency rules affect customer-facing chatbots?"
            }
        ],
        "compliance-agents": [
            {
                label: "Classify airport FR",
                prompt: "Classify a facial recognition system deployed in an airport."
            },
            {
                label: "Credit scoring checklist",
                prompt: "Generate a compliance checklist for an AI credit scoring tool."
            },
            {
                label: "Medical diagnostics docs",
                prompt: "What documentation is required for high-risk medical diagnostics AI?"
            }
        ]
    };

    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({behavior: "smooth"});
    }, [messages, loading]);

    // Reset when changing modes
    useEffect(() => {
        if (mode === "chat") {
            setMessages([{role: "assistant", content: "Hi! Ask me anything about the EU AI Act."}]);
        } else if (mode === "compliance-agents") {
            setMessages([{
                role: "assistant",
                content: "I'll classify your AI system and generate a compliance checklist. Please describe your AI system."
            }]);
        }
        setInput("");
    }, [mode]);

    async function send(overrideText?: string) {
        const rawText = overrideText ?? input;
        const text = rawText.trim();
        if (!text || loading) return;

        const next: Msg[] = [...messages, {role: "user" as const, content: text}];
        setMessages(next);
        setInput("");
        setLoading(true);

        try {
            if (mode === "chat") {
                await sendChat(text, next);
            } else if (mode === "compliance-agents") {
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
                            console.log(`${stepTitle} - Received final_result:`, data.data);
                            finalData = data.data;
                            // Don't create message here - we'll create it after streaming with the formatted content
                        } else if (data.type === "error") {
                            throw new Error(data.message);
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }

            console.log(`${stepTitle} - Streaming complete. ResponseIndex: ${responseMessageIndex}, FinalData:`, finalData);
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

            // Convert relevant_articles to sources (preserve main functionality)
            let classifySources: Source[] | undefined = undefined;
            if (classifyData.relevant_articles && classifyData.relevant_articles.length > 0) {
                classifySources = classifyData.relevant_articles.map((article: string, idx: number) => ({
                    id: `article-${idx}`,
                    content: `Referenced in classification: ${article}`,
                    metadata: { id: article, type: 'article' }
                }));
            }

            // Replace streaming message with formatted Step 1
            if (classifyMsgIndex !== -1) {
                setMessages(m => {
                    const newMessages = [...m];
                    newMessages[classifyMsgIndex] = {
                        ...newMessages[classifyMsgIndex],
                        content: step1Response,
                        sources: classifySources,  // Add sources here
                        metadata: classifyData
                    };
                    return newMessages;
                });
            } else {
                // No streaming message was created (no token events), create new message directly
                console.log("Creating classification message directly (no token streaming)");
                setMessages(m => [...m, {role: "assistant", content: step1Response, sources: classifySources, metadata: classifyData}]);
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

            // Collect all unique articles from checklist items (preserve main functionality)
            const allArticles = new Set<string>();
            const articleToRequirements = new Map<string, string[]>();

            checklistData.checklist_items.forEach((item: any, idx: number) => {
                step2Response += `**${idx + 1}. ${item.requirement}**\n`;
                step2Response += `   - Priority: ${item.priority}\n`;
                step2Response += `   - Category: ${item.category}\n`;
                if (item.applicable_articles && item.applicable_articles.length > 0) {
                    step2Response += `   - Articles: ${item.applicable_articles.join(", ")}\n`;

                    // Track articles and their requirements
                    item.applicable_articles.forEach((article: string) => {
                        allArticles.add(article);
                        if (!articleToRequirements.has(article)) {
                            articleToRequirements.set(article, []);
                        }
                        articleToRequirements.get(article)!.push(item.requirement);
                    });
                }
                step2Response += `\n`;
            });

            // Convert articles to sources format (preserve main functionality)
            let checklistSources: Source[] | undefined = undefined;
            if (allArticles.size > 0) {
                checklistSources = Array.from(allArticles).map((article, idx) => {
                    const requirements = articleToRequirements.get(article) || [];
                    return {
                        id: `checklist-article-${idx}`,
                        content: `${article} - Applies to: ${requirements.join('; ')}`,
                        metadata: { id: article, type: 'article', requirements }
                    };
                });
            }

            // Replace streaming message with formatted Step 2
            if (checklistMsgIndex !== -1) {
                setMessages(m => {
                    const newMessages = [...m];
                    newMessages[checklistMsgIndex] = {
                        ...newMessages[checklistMsgIndex],
                        content: step2Response,
                        sources: checklistSources,  // Add sources here
                        metadata: checklistData
                    };
                    return newMessages;
                });
            } else {
                // No streaming message was created (no token events), create new message directly
                console.log("Creating checklist message directly (no token streaming)");
                setMessages(m => [...m, {role: "assistant", content: step2Response, sources: checklistSources, metadata: checklistData}]);
            }

        } catch (error: any) {
            throw error;
        }
    }

    const handleCopy = async (content: string, index: number) => {
        if (typeof navigator === "undefined" || !navigator.clipboard) {
            return;
        }
        try {
            await navigator.clipboard.writeText(content);
            setCopiedMessageIndex(index);
            setTimeout(() => {
                setCopiedMessageIndex((current) => (current === index ? null : current));
            }, 2000);
        } catch (error) {
            console.error("Failed to copy response", error);
        }
    };


    return (
        <Vortex
            backgroundColor="#03060d"
            particleCount={600}
            baseHue={220}
            rangeHue={80}
            baseSpeed={0.00002}
            rangeSpeed={0.0003}
            className="h-screen w-full"
        >
        <div className="mx-auto flex h-full w-full max-w-5xl flex-col px-4">
            {/* Header with logo and tabs */}
            <div className="flex items-center gap-6 p-4 border-b">
                <img
                    src="/regulaid_logo.png"
                    alt="RegulAId"
                    className="h-10 w-auto object-contain"
                />
                <Tabs value={mode} onValueChange={(value) => setMode(value as Mode)}>
                    <TabsList>
                        <TabsTrigger value="chat">Q&A Chat</TabsTrigger>
                        <TabsTrigger value="compliance-agents">Compliance Agents</TabsTrigger>
                    </TabsList>
                </Tabs>
            </div>

            {/* Main chat area */}
            <div className="flex-1 flex flex-col min-h-0">
                <div className="flex-1 min-h-0">
                    <ScrollArea className="h-full">
                        <div className="mx-auto max-w-3xl p-4 space-y-3">
                                {messages.map((m, i) => {
                                    const showCopy = !m.taskOnly && m.role === "assistant" && Boolean(m.content?.trim());
                                    return (
                                        <div key={i} className="space-y-2">
                                            <Bubble
                                                role={m.role}
                                                content={m.content}
                                                sources={m.sources}
                                                taskOnly={m.taskOnly}
                                                mode={mode}
                                                tasks={m.tasks}
                                            />
                                            {showCopy && (
                                                <div className="flex items-start gap-3 justify-start">
                                                    <div className="h-8 w-8 flex-shrink-0" aria-hidden="true" />
                                                    <div className="max-w-[80%] flex justify-end">
                                                        <Actions className="justify-end">
                                                            <Action
                                                                label={copiedMessageIndex === i ? "Copied" : "Copy"}
                                                                tooltip={copiedMessageIndex === i ? "Copied" : "Copy response"}
                                                                onClick={() => {
                                                                    void handleCopy(m.content as string, i);
                                                                }}
                                                            >
                                                                {copiedMessageIndex === i ? (
                                                                    <CopySuccessIcon className="size-4 text-emerald-500" />
                                                                ) : (
                                                                    <CopyIcon className="size-4" />
                                                                )}
                                                            </Action>
                                                        </Actions>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            {loading && mode === "chat" && <LoadingBubble/>}
                            <div ref={endRef}/>
                        </div>
                    </ScrollArea>
                </div>

                {/* Input area */}
                <div className="mx-auto w-full max-w-3xl px-4 pb-4 space-y-3">
                    <Suggestions>
                        {suggestionPresets[mode].map(({label, prompt}) => (
                            <Suggestion
                                key={prompt}
                                suggestion={prompt}
                                onClick={() => setInput(prompt)}
                            >
                                {label}
                            </Suggestion>
                        ))}
                    </Suggestions>
                    <PromptInput
                        className="divide-y-0"
                        onSubmit={(e) => {
                        e.preventDefault();
                        send();
                    }}>
                        <div className="relative">
                            <PromptInputTextarea
                                value={input}
                                onChange={(e) => setInput(e.currentTarget.value)}
                                placeholder=" "
                                className="placeholder-transparent"
                                onFocus={() => setInputFocused(true)}
                                onBlur={() => setInputFocused(false)}
                            />
                            <div
                                className={cn(
                                    "pointer-events-none absolute inset-x-3 top-3 text-sm transition-opacity",
                                    input || inputFocused ? "opacity-0" : "opacity-100"
                                )}
                            >
                                <TypingText
                                    text={mode === "chat"
                                        ? [
                                            "Ask about airport facial recognition",
                                            "Clarify biometric identification rules",
                                            "Check chatbot transparency duties"
                                        ]
                                        : [
                                            "Classify your AI system",
                                            "Request a compliance checklist",
                                            "List documents for high-risk AI"
                                        ]}
                                    typingSpeed={75}
                                    pauseDuration={1500}
                                    showCursor={true}
                                    cursorCharacter="|"
                                    className="text-muted-foreground"
                                    textColors={['#3b82f6', '#8b5cf6', '#06b6d4']}
                                    variableSpeed={{ min: 50, max: 120 }}
                                />
                            </div>
                        </div>
                        <PromptInputToolbar className="justify-end">
                            <PromptInputSubmit disabled={loading || !input.trim()} />
                        </PromptInputToolbar>
                    </PromptInput>
                </div>
            </div>
        </div>
        </Vortex>
    );

    function LoadingBubble() {
        // Only used for chat mode (non-streaming)
        return (
            <div className="mb-4 flex items-start gap-3 justify-start">
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
                        role, content, muted, sources, taskOnly, mode, tasks
                    }: {
        role: "user" | "assistant";
        content: string;
        muted?: boolean;
        sources?: Source[];
        taskOnly?: boolean;
        mode?: Mode;
        tasks?: Task[];
    }) {
        const isUser = role === "user";

        // If taskOnly is true, only render the Task component
        if (taskOnly && !isUser) {
            return (
                <div className="flex items-start gap-3 justify-start">
                    <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>
                    <div className="max-w-[80%] rounded-2xl px-3 py-2 text-sm bg-muted">
                        {/* Render dynamic tasks from streaming */}
                        <Task>
                            <TaskTrigger title={mode === "compliance-agents" ? "Running full compliance workflow" : "Processing your request"}/>
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
            <div className={`flex items-start gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
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
                                <InlineCitationCard>
                                    <InlineCitationCardTrigger sources={["http://eu-ai-act"]}>
                                        <InlineCitation>
                                            <InlineCitationText logo="/eu-aia-ct.png">
                                                Sources
                                            </InlineCitationText>
                                        </InlineCitation>
                                    </InlineCitationCardTrigger>
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
                                                            title={`${src.metadata?.id || 'EU AI ACT'}`}
                                                            url="#"
                                                            description={src.content}
                                                        />
                                                    </InlineCitationCarouselItem>
                                                ))}
                                            </InlineCitationCarouselContent>
                                        </InlineCitationCarousel>
                                    </InlineCitationCardBody>
                                </InlineCitationCard>
                            )}
                        </>
                    )}
                </div>
                {isUser && <Avatar className="h-8 w-8"><AvatarFallback>U</AvatarFallback></Avatar>}
            </div>
        );
    }
}

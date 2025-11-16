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
    TaskItemFile,
} from '@/components/ui/shadcn-io/ai/task';
import {
    Tool,
    ToolHeader,
    ToolContent,
    ToolInput,
    ToolOutput,
} from "@/components/ui/shadcn-io/ai/tool";

type Source = { id: number | string; content: string; metadata?: Record<string, any> };
type ToolCall = {
    id: string;
    name: string;
    input: any;
    output?: string;
    state: "input-streaming" | "input-available" | "output-available" | "output-error";
    step?: "classification" | "checklist";  // Track which step this tool belongs to
};
type TaskInfo = {
    title: string;
    description?: string;
    status: "in_progress" | "completed";
    items?: string[];
    tools?: ToolCall[];  // Tools shown inside the task
    currentStep?: string;  // Current step name
    // Track which step each tool belongs to
    toolSteps?: Map<string, "classification" | "checklist">;  // Map tool ID to step
};

type Msg = {
    role: "user" | "assistant";
    content: string;
    sources?: Source[];
    metadata?: any;
    tools?: ToolCall[];  // Tool calls for transparency
    isProcessing?: boolean;  // Step is in progress
    task?: TaskInfo;  // Task information for Task component
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
        // Call the unified workflow endpoint and render events with tool visualization
        // Note: user message is already added in send() function, so messages.length is the user message index + 1
        // The task should be inserted right after the user message (at messages.length)

        const res = await fetch("/api/compliance/workflow/stream", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ai_system_description: description}),
        });

        if (!res.ok) throw new Error("Failed to start workflow");
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // Track current step to know which message to update
        let classificationMessageIndex = -1;
        let checklistMessageIndex = -1;
        let currentTools: ToolCall[] = [];
        let toolIdCounter = 0;
        let streamingContent = "";
        // Track which step is currently active for tool insertion
        let currentStep: "classification" | "checklist" | null = null;
        // Track the task message index (right after user message)
        // User message is at messages.length - 1, so task goes at messages.length
        let taskMessageIndex = -1;
        // Track which step each tool belongs to
        const toolStepMap = new Map<string, "classification" | "checklist">();

        try {
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, {stream: true});
                const lines = buffer.split('\n');
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim() || !line.startsWith('data: ')) continue;
                    const event = JSON.parse(line.substring(6));


                    // Handle different event types
                    if (event.type === "step_start") {
                        // Step starting - update task with step info
                        const stepName = event.step === "classification" ? "Classification" : "Checklist Generation";
                        const agentName = event.step === "classification" ? "classification_agent" : "checklist_agent";
                        currentTools = [];  // Reset tools for new step
                        streamingContent = "";  // Reset streaming content
                        currentStep = event.step;  // Track which step is active

                        // Create or update task message right after user message
                        setMessages(m => {
                            const newMessages = [...m];
                            
                            // Calculate task message index (right after last user message)
                            // Find the last user message index
                            let lastUserIndex = -1;
                            for (let i = newMessages.length - 1; i >= 0; i--) {
                                if (newMessages[i].role === "user") {
                                    lastUserIndex = i;
                                    break;
                                }
                            }
                            
                            // Task should be right after the user message
                            const targetTaskIndex = lastUserIndex >= 0 ? lastUserIndex + 1 : newMessages.length;
                            
                            // Create task message if it doesn't exist
                            if (taskMessageIndex === -1 || taskMessageIndex >= newMessages.length) {
                                taskMessageIndex = targetTaskIndex;
                                // Insert right after user message
                                newMessages.splice(taskMessageIndex, 0, {
                                    role: "assistant",
                                    content: "",
                                    task: {
                                        title: "Compliance Workflow",
                                        status: "in_progress",
                                        items: [`Using ${agentName}`],
                                        tools: [],
                                        currentStep: stepName
                                    },
                                    isProcessing: true
                                });
                                // Update other indices
                                if (classificationMessageIndex >= taskMessageIndex) classificationMessageIndex++;
                                if (checklistMessageIndex >= taskMessageIndex) checklistMessageIndex++;
                            } else {
                                // Update existing task - add step item only if it doesn't exist
                                const taskMsg = newMessages[taskMessageIndex];
                                const existingItems = taskMsg.task?.items || [];
                                const stepItem = `Using ${agentName}`;
                                
                                // Only add if not already present
                                if (!existingItems.includes(stepItem)) {
                                    newMessages[taskMessageIndex] = {
                                        ...taskMsg,
                                        task: {
                                            title: taskMsg.task?.title || "Compliance Workflow",
                                            status: "in_progress",
                                            items: [...existingItems, stepItem],
                                            tools: taskMsg.task?.tools || [],
                                            currentStep: stepName
                                        }
                                    };
                                } else {
                                    // Just update current step
                                    newMessages[taskMessageIndex] = {
                                        ...taskMsg,
                                        task: {
                                            ...taskMsg.task!,
                                            currentStep: stepName
                                        }
                                    };
                                }
                            }
                            
                            // Create step message for content
                            const newIndex = newMessages.length;
                            if (event.step === "classification") classificationMessageIndex = newIndex;
                            else checklistMessageIndex = newIndex;
                            newMessages.push({
                                role: "assistant",
                                content: `**${stepName}**\n\nAnalyzing...`,
                                isProcessing: true
                            });
                            
                            return newMessages;
                        });

                    } else if (event.type === "tool_start") {
                        // Tool started - add to task component
                        const newTool: ToolCall = {
                            id: `tool-${toolIdCounter++}`,
                            name: event.tool_name,
                            input: event.tool_input || {},
                            state: "input-available",  // Tool is ready to execute
                            step: currentStep || undefined  // Track which step this tool belongs to
                        };
                        currentTools.push(newTool);
                        
                        // Track which step this tool belongs to
                        if (currentStep) {
                            toolStepMap.set(newTool.id, currentStep);
                        }
                        

                        // Update task message with new tool
                        setMessages(m => {
                            const newMessages = [...m];
                            if (taskMessageIndex >= 0 && taskMessageIndex < newMessages.length) {
                                const taskMsg = newMessages[taskMessageIndex];
                                const existingTools = taskMsg.task?.tools || [];
                                
                                // Insert tool in the right position based on step
                                // Classification tools should come before checklist tools
                                let insertIndex = existingTools.length;
                                if (currentStep === "checklist") {
                                    // Find the first checklist tool or append at end
                                    for (let i = 0; i < existingTools.length; i++) {
                                        const toolStep = toolStepMap.get(existingTools[i].id);
                                        if (toolStep === "checklist") {
                                            insertIndex = i;
                                            break;
                                        }
                                    }
                                } else if (currentStep === "classification") {
                                    // Find where classification tools end (before first checklist tool)
                                    for (let i = 0; i < existingTools.length; i++) {
                                        const toolStep = toolStepMap.get(existingTools[i].id);
                                        if (toolStep === "checklist") {
                                            insertIndex = i;
                                            break;
                                        }
                                    }
                                }
                                
                                const updatedTools = [...existingTools];
                                updatedTools.splice(insertIndex, 0, newTool);
                                
                                newMessages[taskMessageIndex] = {
                                    ...taskMsg,
                                    task: {
                                        ...taskMsg.task!,
                                        tools: updatedTools,
                                        status: "in_progress"
                                    }
                                };
                            }
                            return newMessages;
                        });

                    } else if (event.type === "tool_complete") {
                        // Tool completed - update tool in task component

                        setMessages(m => {
                            const newMessages = [...m];
                            if (taskMessageIndex >= 0 && taskMessageIndex < newMessages.length) {
                                const taskMsg = newMessages[taskMessageIndex];
                                const tools = taskMsg.task?.tools || [];
                                
                                // Find and update the tool
                                const toolIndex = tools.findIndex(t => t.name === event.tool_name && !t.output);
                                if (toolIndex !== -1) {
                                    tools[toolIndex].output = event.tool_output || "";
                                    tools[toolIndex].state = "output-available";
                                } else {
                                    // Tool not found - add it (tool_start was likely missed)
                                    tools.push({
                                        id: `tool-${toolIdCounter++}`,
                                        name: event.tool_name,
                                        input: event.tool_input || {},
                                        output: event.tool_output || "",
                                        state: "output-available",
                                        step: currentStep || undefined  // Track which step this tool belongs to
                                    });
                                }
                                
                                newMessages[taskMessageIndex] = {
                                    ...taskMsg,
                                    task: {
                                        ...taskMsg.task!,
                                        tools: [...tools]
                                    }
                                };
                            }
                            return newMessages;
                        });

                    } else if (event.type === "task") {
                        // Task status update - update the task message at the top
                        setMessages(m => {
                            const newMessages = [...m];
                            
                            // Calculate task message index (right after last user message)
                            let lastUserIndex = -1;
                            for (let i = newMessages.length - 1; i >= 0; i--) {
                                if (newMessages[i].role === "user") {
                                    lastUserIndex = i;
                                    break;
                                }
                            }
                            const targetTaskIndex = lastUserIndex >= 0 ? lastUserIndex + 1 : newMessages.length;
                            
                            // Create task message if it doesn't exist (right after user message)
                            if (taskMessageIndex === -1 || taskMessageIndex >= newMessages.length) {
                                taskMessageIndex = targetTaskIndex;
                                // Insert right after user message
                                newMessages.splice(taskMessageIndex, 0, {
                                    role: "assistant",
                                    content: "",
                                    task: {
                                        title: event.title || "Compliance Workflow",
                                        description: event.description,
                                        status: event.status || "in_progress",
                                        items: event.items || [],
                                        tools: [],
                                        currentStep: currentStep === "classification" ? "Classification" : currentStep === "checklist" ? "Checklist Generation" : undefined
                                    },
                                    isProcessing: event.status === "in_progress"
                                });
                                // Update other indices
                                if (classificationMessageIndex >= taskMessageIndex) classificationMessageIndex++;
                                if (checklistMessageIndex >= taskMessageIndex) checklistMessageIndex++;
                            } else {
                                // Update existing task message
                                const taskMsg = newMessages[taskMessageIndex];
                                const existingItems = taskMsg.task?.items || [];
                                const existingTools = taskMsg.task?.tools || [];
                                
                                newMessages[taskMessageIndex] = {
                                    ...taskMsg,
                                    task: {
                                        title: event.title || taskMsg.task?.title || "Compliance Workflow",
                                        description: event.description || taskMsg.task?.description,
                                        status: event.status || "in_progress",
                                        items: event.items 
                                            ? [...existingItems, ...event.items]
                                            : existingItems,
                                        tools: existingTools,
                                        currentStep: taskMsg.task?.currentStep
                                    },
                                    isProcessing: event.status === "in_progress"
                                };
                            }
                            return newMessages;
                        });

                    } else if (event.type === "content_stream") {
                        // Stream LLM content chunks
                        streamingContent += event.content;
                        
                        // Update current message with streaming content
                        const currentIndex = checklistMessageIndex !== -1 && checklistMessageIndex > classificationMessageIndex
                            ? checklistMessageIndex
                            : classificationMessageIndex;

                        setMessages(m => {
                            const newMessages = [...m];
                            if (currentIndex !== -1) {
                                const currentMsg = newMessages[currentIndex];
                                // Get base content (remove "Analyzing..." placeholder)
                                const baseContent = (currentMsg.content || "").replace(/Analyzing\.\.\./g, "").replace(/\*\*.*?\*\*\n\n/g, "");
                                const stepName = classificationMessageIndex === currentIndex ? "Classification" : "Checklist Generation";
                                newMessages[currentIndex] = {
                                    ...currentMsg,
                                    content: `**${stepName}**\n\n${baseContent}${streamingContent}`,
                                    isProcessing: true
                                };
                            }
                            return newMessages;
                        });

                    } else if (event.type === "step_complete") {
                        // Step complete - format and display result
                        const {step, data} = event;
                        streamingContent = "";  // Reset streaming content
                        currentTools = [];  // Clear tools array

                        // Update task status if checklist is complete (workflow done)
                        if (step === "checklist") {
                            setMessages(m => {
                                const newMessages = [...m];
                                if (taskMessageIndex >= 0 && taskMessageIndex < newMessages.length) {
                                    const taskMsg = newMessages[taskMessageIndex];
                                    newMessages[taskMessageIndex] = {
                                        ...taskMsg,
                                        task: {
                                            ...taskMsg.task!,
                                            status: "completed"
                                        }
                                    };
                                }
                                return newMessages;
                            });
                        }

                        if (step === "classification") {
                            const formatted = formatClassification(data);
                            const sources = extractClassificationSources(data);

                            setMessages(m => {
                                const newMessages = [...m];
                                // Update the step message
                                newMessages[classificationMessageIndex] = {
                                    role: "assistant",
                                    content: formatted,
                                    sources,
                                    metadata: data,
                                    isProcessing: false
                                };
                                return newMessages;
                            });

                        } else if (step === "checklist") {
                            const formatted = formatChecklist(data);
                            const sources = extractChecklistSources(data);

                            setMessages(m => {
                                const newMessages = [...m];
                                // Update the step message
                                newMessages[checklistMessageIndex] = {
                                    role: "assistant",
                                    content: formatted,
                                    sources,
                                    metadata: data,
                                    isProcessing: false
                                };
                                return newMessages;
                            });
                        }

                    } else if (event.type === "workflow_pause") {
                        // Workflow paused - needs more info
                        const response = `I need more information before I can classify your system.\n\n**Questions:**\n${event.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}\n\nPlease provide more details.`;

                        setMessages(m => {
                            const newMessages = [...m];
                            newMessages[classificationMessageIndex] = {
                                role: "assistant",
                                content: response,
                                isProcessing: false
                            };
                            return newMessages;
                        });

                    } else if (event.type === "error") {
                        throw new Error(event.message);
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    // Helper functions to format results
    function formatClassification(data: any): string {
        let result = `**Classification Results**\n\n`;
        result += `**Risk Level:** ${data.risk_level}\n`;
        result += `**System Type:** ${data.system_type}\n`;
        result += `**Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;
        result += `**Reasoning:** ${data.reasoning}\n\n`;
        if (data.relevant_articles && data.relevant_articles.length > 0) {
            result += `**Relevant Articles:** ${data.relevant_articles.join(", ")}`;
        }
        return result;
    }

    function extractClassificationSources(data: any): Source[] | undefined {
        if (!data.relevant_articles || data.relevant_articles.length === 0) return undefined;
        return data.relevant_articles.map((article: string, idx: number) => ({
            id: `article-${idx}`,
            content: `Referenced in classification: ${article}`,
            metadata: { id: article, type: 'article' }
        }));
    }

    function formatChecklist(data: any): string {
        let result = `**Compliance Checklist**\n\n`;
        result += `**Total Items:** ${data.total_items}\n\n`;
        result += `${data.summary}\n\n`;
        result += `**Requirements:**\n\n`;

        data.checklist_items.forEach((item: any, idx: number) => {
            result += `**${idx + 1}. ${item.requirement}**\n`;
            result += `   - Priority: ${item.priority}\n`;
            result += `   - Category: ${item.category}\n`;
            if (item.applicable_articles && item.applicable_articles.length > 0) {
                result += `   - Articles: ${item.applicable_articles.join(", ")}\n`;
            }
            result += `\n`;
        });

        return result;
    }

    function extractChecklistSources(data: any): Source[] | undefined {
        const allArticles = new Set<string>();
        const articleToRequirements = new Map<string, string[]>();

        data.checklist_items.forEach((item: any) => {
            if (item.applicable_articles) {
                item.applicable_articles.forEach((article: string) => {
                    allArticles.add(article);
                    if (!articleToRequirements.has(article)) {
                        articleToRequirements.set(article, []);
                    }
                    articleToRequirements.get(article)!.push(item.requirement);
                });
            }
        });

        if (allArticles.size === 0) return undefined;

        return Array.from(allArticles).map((article, idx) => {
            const requirements = articleToRequirements.get(article) || [];
            return {
                id: `checklist-article-${idx}`,
                content: `${article} - Applies to: ${requirements.join('; ')}`,
                metadata: { id: article, type: 'article', requirements }
            };
        });
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
        <div className="min-h-screen w-full flex flex-col bg-gradient-to-b from-slate-50 via-white to-slate-100">
        <div className="flex min-h-screen w-full flex-col px-4 py-8">
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
                                    const showCopy = m.role === "assistant" && Boolean(m.content?.trim()) && !m.isProcessing;
                                    return (
                                        <div key={i} className="space-y-2">
                                            <Bubble
                                                role={m.role}
                                                content={m.content}
                                                sources={m.sources}
                                                tools={m.tools}
                                                task={m.task}
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
        </div>
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
                        role, content, sources, tools, task
                    }: {
        role: "user" | "assistant";
        content: string;
        sources?: Source[];
        tools?: ToolCall[];
        task?: TaskInfo;
    }) {
        const isUser = role === "user";
        const isToolOnly = !isUser && tools && tools.length > 0 && (!content || !content.trim());


        // Render tool-only messages without bubble/avatar
        if (isToolOnly) {
            return (
                <div className="w-full">
                    {tools.map((tool) => {
                        const toolType = tool.name.startsWith('tool-') ? tool.name : `tool-${tool.name}`;
                        
                        return (
                            <Tool key={tool.id} defaultOpen={false}>
                                <ToolHeader
                                    type={toolType as any}
                                    state={tool.state}
                                />
                                <ToolContent>
                                    <ToolInput input={tool.input} />
                                    {((tool.state === 'input-streaming') || (tool.state === 'input-available' && !tool.output)) && (
                                        <div className="mt-3 text-sm text-muted-foreground animate-pulse">
                                            {tool.state === 'input-streaming' ? 'Receiving parameters...' : 'Executing tool...'}
                                        </div>
                                    )}
                                    {tool.output && tool.state === 'output-available' && (
                                        <ToolOutput
                                            output={
                                                <Response>
                                                    {typeof tool.output === 'string' 
                                                        ? (tool.output.length > 2000 
                                                            ? tool.output.substring(0, 2000) + '\n\n... (truncated, showing first 2000 characters)' 
                                                            : tool.output)
                                                        : (typeof tool.output === 'object'
                                                            ? JSON.stringify(tool.output, null, 2)
                                                            : String(tool.output))}
                                                </Response>
                                            }
                                            errorText={tool.state === "output-error" ? "Tool execution failed" : undefined}
                                        />
                                    )}
                                </ToolContent>
                            </Tool>
                        );
                    })}
                </div>
            );
        }

        // Regular message with bubble
        return (
            <div className={`flex items-start gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
                {!isUser && <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>}
                <div
                    className={[
                        "max-w-[80%] rounded-2xl px-3 py-2 text-sm space-y-3",
                        isUser ? "bg-primary text-primary-foreground" : "bg-muted",
                    ].join(" ")}
                >
                    {isUser ? (
                        <span className="whitespace-pre-wrap break-words break-normal">{content}</span>
                    ) : (
                        <>
                            {/* Render Task component if task info is available */}
                            {task && (
                                <div className={content && content.trim() ? "mb-3" : ""}>
                                    <Task>
                                        <TaskTrigger title={task.title} />
                                        <TaskContent>
                                            {/* Show current step */}
                                            {task.currentStep && (
                                                <TaskItem>
                                                    Current step: <TaskItemFile>{task.currentStep}</TaskItemFile>
                                                </TaskItem>
                                            )}
                                            
                                            {/* Show step items and their associated tools grouped together */}
                                            {task.items && task.items.length > 0 && (
                                                task.items.map((item, idx) => {
                                                    const isClassification = item.includes("classification_agent");
                                                    const isChecklist = item.includes("checklist_agent");
                                                    
                                                    // Get tools for this step using the step property
                                                    const relevantTools = task.tools?.filter(tool => {
                                                        if (isClassification) {
                                                            return tool.step === "classification";
                                                        } else if (isChecklist) {
                                                            return tool.step === "checklist";
                                                        }
                                                        return false;
                                                    }) || [];
                                                    
                                                    return (
                                                        <div key={`item-${idx}`} className="space-y-2">
                                                            <TaskItem>{item}</TaskItem>
                                                            {/* Show tools for this step */}
                                                            {relevantTools.length > 0 && (
                                                                <div className="space-y-2 ml-4">
                                                                    {relevantTools.map((tool, toolIdx) => {
                                                                        const toolType = tool.name.startsWith('tool-') ? tool.name : `tool-${tool.name}`;
                                                                        return (
                                                                            <Tool key={`tool-${tool.id || toolIdx}`} defaultOpen={false}>
                                                                                <ToolHeader
                                                                                    type={toolType as any}
                                                                                    state={tool.state}
                                                                                />
                                                                                <ToolContent>
                                                                                    <ToolInput input={tool.input} />
                                                                                    {((tool.state === 'input-streaming') || (tool.state === 'input-available' && !tool.output)) && (
                                                                                        <div className="mt-3 text-sm text-muted-foreground animate-pulse">
                                                                                            {tool.state === 'input-streaming' ? 'Receiving parameters...' : 'Executing tool...'}
                                                                                        </div>
                                                                                    )}
                                                                                    {tool.output && tool.state === 'output-available' && (
                                                                                        <ToolOutput
                                                                                            output={
                                                                                                <Response>
                                                                                                    {typeof tool.output === 'string' 
                                                                                                        ? (tool.output.length > 2000 
                                                                                                            ? tool.output.substring(0, 2000) + '\n\n... (truncated, showing first 2000 characters)' 
                                                                                                            : tool.output)
                                                                                                        : (typeof tool.output === 'object'
                                                                                                            ? JSON.stringify(tool.output, null, 2)
                                                                                                            : String(tool.output))}
                                                                                                </Response>
                                                                                            }
                                                                                            errorText={tool.state === "output-error" ? "Tool execution failed" : undefined}
                                                                                        />
                                                                                    )}
                                                                                </ToolContent>
                                                                            </Tool>
                                                                        );
                                                                    })}
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })
                                            )}
                                            
                                            {/* Show description if no items or tools */}
                                            {(!task.items || task.items.length === 0) && 
                                             (!task.tools || task.tools.length === 0) && 
                                             task.description && (
                                                <TaskItem>{task.description}</TaskItem>
                                            )}
                                        </TaskContent>
                                    </Task>
                                </div>
                            )}

                            {/* Only show Response if there's actual content (not just "Analyzing..." or empty) */}
                            {content && content.trim() && !content.match(/^(\*\*.*?\*\*)?\s*\n?\s*Analyzing\.\.\.?\s*$/i) && (
                                <Response>{content}</Response>
                            )}

                            {/* Render sources */}
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

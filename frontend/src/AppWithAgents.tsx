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

type Source = { id: number | string; content: string; metadata?: Record<string, any> };
type Msg = { role: "user" | "assistant"; content: string; sources?: Source[]; metadata?: any };
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
        setMessages(m => [...m, {role: "assistant", content: answer, sources}]);
    }

    async function sendClassify(description: string) {
        const res = await fetch("/api/classify/", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ai_system_description: description}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(JSON.stringify(data));

        setClassificationResult(data);

        // Format response
        let response = "";
        let sources: Source[] | undefined = undefined;

        if (data.needs_more_info) {
            response = `I need more information to classify your system.\n\n**Questions:**\n${data.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}`;
        } else {
            response = `**Classification Result**\n\n`;
            response += `**Risk Level:** ${data.risk_level}\n`;
            response += `**System Type:** ${data.system_type}\n`;
            response += `**Confidence:** ${(data.confidence * 100).toFixed(0)}%\n\n`;
            response += `**Reasoning:** ${data.reasoning}`;

            // Convert relevant_articles to sources format
            if (data.relevant_articles && data.relevant_articles.length > 0) {
                sources = data.relevant_articles.map((article: string, idx: number) => ({
                    id: `article-${idx}`,
                    content: `Referenced in classification: ${article}`,
                    metadata: { id: article, type: 'article' }
                }));
            }
        }

        setMessages(m => [...m, {role: "assistant", content: response, sources, metadata: data}]);
    }

    async function sendFullFlow(description: string) {
        // Step 1: Classify
        setMessages(m => [...m, {role: "assistant", content: "Step 1: Classifying your AI system..."}]);

        const classifyRes = await fetch("/api/classify/", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ai_system_description: description}),
        });
        const classifyData = await classifyRes.json();
        if (!classifyRes.ok) throw new Error(JSON.stringify(classifyData));

        if (classifyData.needs_more_info) {
            let response = `I need more information before I can classify your system.\n\n**Questions:**\n${classifyData.questions.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n')}\n\nPlease provide more details.`;
            setMessages(m => {
                const newMessages = [...m];
                newMessages[newMessages.length - 1] = {role: "assistant", content: response};
                return newMessages;
            });
            return;
        }

        // Show classification
        let classifyResponse = `**Classification Complete**\n\n`;
        classifyResponse += `**Risk Level:** ${classifyData.risk_level}\n`;
        classifyResponse += `**System Type:** ${classifyData.system_type}\n`;
        classifyResponse += `**Confidence:** ${(classifyData.confidence * 100).toFixed(0)}%\n\n`;
        classifyResponse += `**Reasoning:** ${classifyData.reasoning}`;

        // Convert relevant_articles to sources
        let classifySources: Source[] | undefined = undefined;
        if (classifyData.relevant_articles && classifyData.relevant_articles.length > 0) {
            classifySources = classifyData.relevant_articles.map((article: string, idx: number) => ({
                id: `article-${idx}`,
                content: `Referenced in classification: ${article}`,
                metadata: { id: article, type: 'article' }
            }));
        }

        setMessages(m => {
            const newMessages = [...m];
            newMessages[newMessages.length - 1] = {role: "assistant", content: classifyResponse, sources: classifySources};
            return newMessages;
        });

        // Step 2: Generate checklist
        setMessages(m => [...m, {role: "assistant", content: "Step 2: Generating compliance checklist..."}]);

        const checklistRes = await fetch("/api/checklist/", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                risk_level: classifyData.risk_level,
                system_type: classifyData.system_type,
                system_description: description
            }),
        });
        const checklistData = await checklistRes.json();
        if (!checklistRes.ok) throw new Error(JSON.stringify(checklistData));

        // Format checklist
        let checklistResponse = `**Compliance Checklist (${checklistData.total_items} items)**\n\n`;
        checklistResponse += `${checklistData.summary}\n\n`;
        checklistResponse += `**Requirements:**\n\n`;

        // Collect all unique articles from checklist items
        const allArticles = new Set<string>();
        const articleToRequirements = new Map<string, string[]>();

        checklistData.checklist_items.forEach((item: any, idx: number) => {
            checklistResponse += `**${idx + 1}. ${item.requirement}**\n`;
            checklistResponse += `   - Priority: ${item.priority}\n`;
            checklistResponse += `   - Category: ${item.category}\n`;
            if (item.applicable_articles && item.applicable_articles.length > 0) {
                checklistResponse += `   - Articles: ${item.applicable_articles.join(", ")}\n`;

                // Track articles and their requirements
                item.applicable_articles.forEach((article: string) => {
                    allArticles.add(article);
                    if (!articleToRequirements.has(article)) {
                        articleToRequirements.set(article, []);
                    }
                    articleToRequirements.get(article)!.push(item.requirement);
                });
            }
            checklistResponse += `\n`;
        });

        // Convert articles to sources format
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

        setMessages(m => {
            const newMessages = [...m];
            newMessages[newMessages.length - 1] = {role: "assistant", content: checklistResponse, sources: checklistSources, metadata: checklistData};
            return newMessages;
        });
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
                {/* Logo */}
                <div className="flex items-center gap-3">
                    <img
                        src="/regulaid_logo.png"
                        alt="RegulAId"
                        className="h-12 w-auto object-contain"
                    />
                </div>

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

                <Card className="h-[calc(80vh-7rem)] grid grid-rows-[1fr_auto] overflow-hidden">
                    <div className="min-h-0">
                        <ScrollArea className="h-full">
                            <div className="p-4 space-y-3">
                                {messages.map((m, i) => <Bubble key={i} role={m.role} content={m.content}
                                                                sources={m.sources}/>)}
                                {loading && <Bubble role="assistant" content="…thinking" muted/>}
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

    function Bubble({
                        role, content, muted, sources,
                    }: { role: "user" | "assistant"; content: string; muted?: boolean; sources?: Source[] }) {
        const isUser = role === "user";

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
                                        <InlineCitationCardTrigger sources={["http://eu-ai-act"]}/>
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

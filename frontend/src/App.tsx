import {useEffect, useRef, useState} from "react";
import {Card} from "@/components/ui/card";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Input} from "@/components/ui/input";
import {Button} from "@/components/ui/button";
import {Avatar, AvatarFallback} from "@/components/ui/avatar";
import {Response} from "@/components/ui/shadcn-io/ai/response";
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
type Msg = { role: "user" | "assistant"; content: string; sources?: Source[] };
type ApiResponse = { result?: string; sources?: Source[] } | string[] | string | unknown;

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
    const [messages, setMessages] = useState<Msg[]>([
        {role: "assistant", content: "Hi! Ask me anything."},
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({behavior: "smooth"});
    }, [messages, loading]);

    async function send() {
        const text = input.trim();
        if (!text || loading) return;

        const next: Msg[] = [...messages, {role: "user" as const, content: text}];

        setMessages(next);
        setInput("");
        setLoading(true);

        try {
            const history = next.map(m => m.content);
            const res = await fetch("/api/search/", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query: text, history}),
            });
            const data: ApiResponse = await res.json();
            if (!res.ok) throw new Error(toText(data));
            const answer = toText(data);
            const sources = toSources(data);
            setMessages(m => [...m, {role: "assistant", content: answer, sources}]);
        } catch (e: any) {
            setMessages(m => [...m, {role: "assistant", content: String(e?.message ?? e)}]);
        } finally {
            setLoading(false);
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
            <div className="mx-auto max-w-3xl h-full min-h-0">
                <Card className="h-[80vh] grid grid-rows-[1fr_auto] overflow-hidden">
                    {/* row 1 must be min-h-0 so the viewport can scroll */}
                    <div className="min-h-0">
                        <ScrollArea className="h-full">
                            <div className="p-4 space-y-3">
                                {messages.map((m, i) => <Bubble key={i} role={m.role} content={m.content}
                                                                sources={m.sources}/>)}
                                {loading && <LoadingBubble/>}
                                <div ref={endRef}/>
                            </div>
                        </ScrollArea>
                    </div>

                    <div className="border-t p-3 flex gap-2">
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={onKey}
                            placeholder="Type and press Enter…"
                        />
                        <Button onClick={send} disabled={loading || !input.trim()}>Send</Button>
                    </div>
                </Card>
            </div>
        </div>
    );

    function LoadingBubble() {
        return (
            <div className="mb-1 flex items-start gap-3 justify-start">
                <Avatar className="h-8 w-8"><AvatarFallback>AI</AvatarFallback></Avatar>
                <div className="max-w-[80%] rounded-2xl px-3 py-2 text-sm bg-muted">
                    <Task>
                        <TaskTrigger title="Processing your request" />
                        <TaskContent>
                            <TaskItem>Analyzing your question</TaskItem>
                            <TaskItem>
                                Searching through <TaskItemFile>EU AI Act documents</TaskItemFile>
                            </TaskItem>
                            <TaskItem>
                                Reading <TaskItemFile>regulations.json</TaskItemFile>
                            </TaskItem>
                            <TaskItem>Generating comprehensive response</TaskItem>
                        </TaskContent>
                    </Task>
                </div>
            </div>
        );
    }

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
                            )}
                        </>
                    )}
                </div>
                {isUser && <Avatar className="h-8 w-8"><AvatarFallback>U</AvatarFallback></Avatar>}
            </div>
        );
    }
}

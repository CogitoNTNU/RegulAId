import React, { useState, useRef, useEffect } from "react";

type Msg = { role: "user" | "assistant"; content: string };

type SearchResult = { chunk_id: string; text: string; score?: number | null };
type SearchResponse = { result: string };

export default function App() {
    const [messages, setMessages] = useState<Msg[]>([
        { role: "assistant", content: "Hi! Ask me anything about your data." }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const listRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    }, [messages, loading]);

    async function send() {
        const text = input.trim();
        if (!text || loading) return;
        setInput("");
        const next = [...messages, { role: "user", content: text }];
        setMessages(next);
        setLoading(true);

        try {
            const history: string[] = messages.map(m => m.content);
            const res = await fetch("/api/search/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: text, history: history })
            });
            const raw = await res.text();                      // robust: accept JSON or plain text
            if (!res.ok) throw new Error(raw || `HTTP ${res.status}`);
            let answer = raw;
            try {
                const data = JSON.parse(raw) as Partial<SearchResponse>;
                if (typeof data.result === "string") answer = data.result;
            } catch { /* keep raw */ }

            setMessages(m => [...m, { role: "assistant", content: answer }]);
        } catch (e: any) {
            setMessages(m => [...m, { role: "assistant", content: `Error: ${e.message || e}` }]);
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
        <div style={{ height: "100%", display: "grid", gridTemplateRows: "1fr auto", fontFamily: "system-ui, sans-serif" }}>
            <div ref={listRef} style={{ overflowY: "auto", padding: "16px", background: "#9d34ffff" }}>
                {messages.map((m, i) => (
                    <div
                        key={i}
                        style={{
                            maxWidth: 820,
                            margin: "0 auto 12px",
                            padding: "12px 14px",
                            borderRadius: 12,
                            whiteSpace: "pre-wrap",
                            color: m.role === "user" ? "#0b1020" : "#e8ebf7",
                            background: m.role === "user" ? "#e8ebf7" : "#f828ffff"
                        }}
                    >
                        <div style={{ opacity: 0.65, fontSize: 12, marginBottom: 4 }}>{m.role}</div>
                        {m.content}
                    </div>
                ))}
                {loading && (
                    <div style={{ maxWidth: 820, margin: "0 auto 12px", padding: "12px 14px", borderRadius: 12, background: "#1a2344", color: "#e8ebf7" }}>
                        <div style={{ opacity: 0.65, fontSize: 12, marginBottom: 4 }}>assistant</div>
                        <span>…thinking</span>
                    </div>
                )}
            </div>

            <div style={{ display: "flex", gap: 8, padding: 12, borderTop: "1px solid #e5e7eb", background: "white" }}>
                <input
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={onKey}
                    placeholder="Type a message and press Enter…"
                    style={{ flex: 1, padding: "12px 14px", borderRadius: 10, border: "1px solid #cbd5e1" }}
                />
                <button
                    onClick={send}
                    disabled={loading || !input.trim()}
                    style={{ padding: "12px 16px", borderRadius: 10, border: "1px solid #0b1020", background: "#0b1020", color: "white", cursor: loading ? "not-allowed" : "pointer" }}
                >
                    Send
                </button>
            </div>
        </div>
    );
}

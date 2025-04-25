// React + Tailwind web app for RaviBot
import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { motion } from "framer-motion";

export default function RaviBotApp() {
  const [messages, setMessages] = useState([
    { sender: "RaviBot", text: "Hi! Ask me anything about Grade 11 History." }
  ]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const askRaviBot = async () => {
    if (!query.trim()) return;
    const userMessage = { sender: "You", text: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();
    const botMessage = { sender: "RaviBot", text: data.answer };
    setMessages((prev) => [...prev, botMessage]);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#090a09] text-white p-4 flex flex-col">
      <h1 className="text-3xl font-bold text-center text-[#798f0a] mb-4">RaviBot 🤖</h1>
      <ScrollArea className="flex-1 overflow-y-auto space-y-3 mb-4 p-2">
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`w-fit max-w-[85%] px-4 py-2 rounded-2xl shadow ${
              msg.sender === "RaviBot"
                ? "bg-[#1d1d1d] self-start"
                : "bg-[#798f0a] text-black self-end"
            }`}
          >
            <strong>{msg.sender}:</strong> {msg.text}
          </motion.div>
        ))}
      </ScrollArea>
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && askRaviBot()}
          className="bg-[#1d1d1d] border border-[#798f0a] text-white"
          placeholder="Ask a history question..."
        />
        <Button onClick={askRaviBot} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </Button>
      </div>
    </div>
  );
}
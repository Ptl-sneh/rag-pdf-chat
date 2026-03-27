import { ChatGroq } from "@langchain/groq";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

export async function buildRAGChain(vectorStore: FaissStore) {

  // 1. Initialize Groq LLM
  const llm = new ChatGroq({
    model: "llama3-8b-8192",
    temperature: 0,
  });

  // 2. Basic retriever — fetches top 4 relevant chunks
  const retriever = vectorStore.asRetriever({ k: 4 });

  // 3. Prompt to rephrase follow-up questions using chat history
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    ["human", "Given the above conversation, rephrase the follow-up question as a standalone question."],
  ]);

  // 4. Wrap retriever with history awareness
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });

  // 5. Main QA prompt — LLM answers from retrieved context
  const qaPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a helpful assistant. Answer the user's question based only on the context below.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

Context:
{context}`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  // 6. Chain that stuffs retrieved docs into the prompt
  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
  });

  // 7. Final RAG chain — retriever + document chain wired together
  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: documentChain,
  });

  return ragChain;
}
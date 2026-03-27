import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"; // Reads the pdf file and converts it into langchain documents.
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"; // Splits documents into smaller chunks
import { OllamaEmbeddings } from "@langchain/ollama"; // Generates embedding of the documents
import { FaissStore } from "@langchain/community/vectorstores/faiss"; // Store the embeddings in a vector store.
import { loggerInfo } from "./logger/loggerInfo";

// 
export const ingestPdf = async (pdfPath: string): Promise<FaissStore> => {
    loggerInfo(`Loading PDF`)
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load(); // .load() returns an array of Document objects — one Document per page

    loggerInfo(`Loaded Document ${docs.length} pages`)

    loggerInfo(`Splitting into the chunks`)


    /*

    Why do we split?

    LLMs have a context window limit — you can't feed an entire 50-page PDF to the model at once. Instead we:
    1. Split the PDF into small chunks (~1000 characters each)
    2. When user asks a question, we find only the relevant chunks and send those

    */
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, // each chunk is size of 1000
        chunkOverlap: 200, // it ensures that while splitting the chunks, we dont lose context.
    })

    const chunks = await splitter.splitDocuments(docs)

    loggerInfo(`Split into ${chunks.length} chunks`)

    loggerInfo("Generating Embeddings via ollama")

    const embeddings = new OllamaEmbeddings({
        model : "nomic-embed-text" // model to convert text --> vectors
    })

    const vectorStore = await FaissStore.fromDocuments(chunks, embeddings) // creates a vector store from the chunks and embeddings

    loggerInfo("Embeddings Generated and stored in Vector Store")

    return vectorStore
}
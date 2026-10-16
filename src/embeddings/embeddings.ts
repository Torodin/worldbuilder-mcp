import {
    type FeatureExtractionPipeline,
    pipeline,
} from "@huggingface/transformers";

let embedder: FeatureExtractionPipeline | null = null;

export async function getEmbedding(text: string): Promise<number[]> {
    if (!embedder)
        embedder = await pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2",
        );

    const output = await embedder(text, {
        pooling: "mean",
        normalize: true,
    });
    return Array.from(output.data);
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
    return vecA.reduce(
        (product, current, index) => product + current * (vecB[index] ?? 0),
        0,
    );
}

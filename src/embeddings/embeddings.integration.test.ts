import { getEmbedding } from "./embeddings.js";

describe("Embeddings integration", () => {
    describe("getEmbeddings", () => {
        it("should return 384-dimensional normalized vector", async () => {
            const embedding = await getEmbedding("sample input");

            // The embeddin all-MiniLM-L6-v2 maps sentences to a 384 dimensional vector
            expect(embedding).toHaveLength(384);

            const squaredSum = embedding.reduce(
                (sum, val) => sum + val * val,
                0,
            );
            const norm = Math.sqrt(squaredSum);
            expect(norm).toBeCloseTo(1.0, 4);
        }, 30000);
    });
});

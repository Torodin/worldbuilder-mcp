import { pipeline } from "@huggingface/transformers";
import { cosineSimilarity, getEmbedding } from "./embeddings.js";

vi.mock("@huggingface/transformers", () => {
    const mockEmbedder = vi.fn().mockResolvedValue({
        data: new Float32Array([0.1, 0.2, 0.3]),
    });

    return {
        pipeline: vi.fn().mockResolvedValue(mockEmbedder),
    };
});

describe("Embedding", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe("getEmbedding", () => {
        it("should initialize the pipeline and return an array", async () => {
            const sampleInput = "Input sample";
            const result = await getEmbedding(sampleInput);

            expect(pipeline).toHaveBeenCalledTimes(1);

            const mockEmbedder =
                await vi.mocked(pipeline).mock.results[0]?.value;
            expect(mockEmbedder).toHaveBeenCalledWith(sampleInput, {
                pooling: "mean",
                normalize: true,
            });

            expect(result).toEqual(
                Array.from(new Float32Array([0.1, 0.2, 0.3])),
            );
            expect(Array.isArray(result)).toBe(true);
        });
    });

    describe("cosineSimilarity", () => {
        it("should return 1 for identical normalized vectors", () => {
            const vecA = [1, 0, 0];
            const vecB = [1, 0, 0];

            expect(cosineSimilarity(vecA, vecB)).toBe(1);
        });

        it("should return 0 for orthogonal vectors", () => {
            const vecA = [1, 0, 0];
            const vecB = [0, 1, 0];

            expect(cosineSimilarity(vecA, vecB)).toBe(0);
        });

        it("should correctly calculate dot product for arbitrary vectors", () => {
            const vecA = [0.5, 0.5, 0];
            const vecB = [0.5, -0.5, 0];

            expect(cosineSimilarity(vecA, vecB)).toBe(0);

            const vecC = [0.1, 0.2, 0.3];
            const vecD = [0.4, 0.5, 0.6];

            expect(cosineSimilarity(vecC, vecD)).toBe(0.32);
        });
    });
});

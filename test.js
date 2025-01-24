import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  AutoTokenizer,
} from "@xenova/transformers";
import { configDotenv } from "dotenv";
import { MongoClient } from "mongodb";
import { Embeddings } from "@langchain/core/embeddings";
import sharp from "sharp";
import path from "path";
import fs from "fs";

configDotenv();

const client = new MongoClient(process.env.MONGO_URI);
await client.connect();
const db = client.db("bot");
const collection = db.collection("image");




let currentImagePath = null;

class CustomEmbeddings extends Embeddings {
  constructor(getEmbeddings) {
    super();
    this.getEmbeddings = getEmbeddings;
  }

  async embedQuery(query) {
    
    return this.getEmbeddings(query);
  }

  async embedDocuments(documents) {
    console.log("Processing documents:", documents);
    const embeddings = await Promise.all(
      documents.map((doc) => this.getEmbeddings(doc.imagePath))
    );
    return embeddings;
  }
}

const getImageEmbedding = async (imagePath) => {
  try {
    const absolutePath = path.resolve(imagePath);
    console.log("Processing image:", absolutePath);

    const data = await sharp(absolutePath).toBuffer();

    const quantized = true;
    const imageProcessor = await AutoProcessor.from_pretrained(
      "Xenova/clip-vit-base-patch32",
      { quantized: true }
    );
    const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
      "Xenova/clip-vit-base-patch32",
      { quantized: true }
    );
    const tokenizer = await AutoTokenizer.from_pretrained(
      "Xenova/clip-vit-base-patch32"
    );

    const processedImage = await sharp(data)
      .resize(224, 224, {
        fit: "contain",
        background: { r: 255, g: 255, b: 255 },
      })
      .withMetadata()
      .toFormat("jpeg", {
        quality: 100,
        chromaSubsampling: "4:4:4",
      })
      .toBuffer();

    const base64Image = Buffer.from(processedImage).toString("base64");
    const byteArray = Buffer.from(base64Image, "base64");

    const blob = new Blob([byteArray], { type: "image/jpeg" });

    let image = await RawImage.fromBlob(blob);
    let imageInputs = await imageProcessor(image);
    let { image_embeds } = await visionModel(imageInputs);

    return Array.from(image_embeds.data);
  } catch (error) {
    console.error("Error processing image:", error);
    throw error;
  }
};

(async () => {
  try {
    const currentDir = process.cwd();
    currentImagePath = path.join(currentDir, "./cookie.png");
    console.log("Query image path:", currentImagePath);

    // Get embeddings for the query image
    const queryEmbeddings = await getImageEmbedding(currentImagePath);

    console.log("Query embeddings:", queryEmbeddings);
    await fs.promises.writeFile("./temp/em.txt", JSON.stringify(queryEmbeddings));
    const cursor = await collection.aggregate([
      {
        $vectorSearch: {
          index: "image_index",
          path: "embedding",
          queryVector: queryEmbeddings,
          numCandidates: 100,
          limit: 1,
        },
      },
      {
        $project: {
          _id: 1,
          text: 1,
          imagePath: 1,
          score: 1,
        },
      },
    ]);

    // Get all results
    const results = await cursor.toArray({
      limit: 3,
    });

    // Log the search results with more detail
    console.log("Similar images found:", results);

    await client.close();
  } catch (error) {
    console.error("Detailed error:", error);
    throw error;
  }
})();

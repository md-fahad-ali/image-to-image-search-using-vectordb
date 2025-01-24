import { pipeline } from "@xenova/transformers";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { configDotenv } from "dotenv";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { Embeddings } from "@langchain/core/embeddings";
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  AutoTokenizer,
} from "@xenova/transformers";

configDotenv();
import readlineSync from "readline-sync";

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
    return this.getEmbeddings(currentImagePath);
  }

  async embedDocuments(documents) {
    const embeddings = await Promise.all(
      documents.map(() => this.getEmbeddings(currentImagePath))
    );
    return embeddings;
  }
}

async function getImageEmbedding(imagePath) {
  try {
    const resolvedPath = path.resolve(imagePath);

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`Image not found: ${resolvedPath}`);
    }

    const processedImage = await sharp(resolvedPath)
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

    // const tempImagePath = path.join(
    //   path.dirname(resolvedPath),
    //   "temp_processed.jpg"
    // );
    // await fs.promises.writeFile(tempImagePath, processedImage);

    // const imageFeatureExtractor = await pipeline(
    //   "image-feature-extraction",
    //   "Xenova/clip-vit-base-patch32"
    // );

    // const features = await imageFeatureExtractor(tempImagePath);

    // await fs.promises.unlink(tempImagePath);
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

    const base64Image = Buffer.from(processedImage).toString("base64");
    const byteArray = Buffer.from(base64Image, "base64");

    const blob = new Blob([byteArray], { type: "image/jpeg" });

    let image = await RawImage.fromBlob(blob);
    let imageInputs = await imageProcessor(image);
    let { image_embeds } = await visionModel(imageInputs);

    return Array.from(image_embeds.data);
  } catch (error) {
    console.error("Embedding Extraction Error:", error);
    throw error;
  }
}

async function main() {
  try {
    // Set the current image path
    // currentImagePath = "./skirt.jpg";
    const input = readlineSync.question("Enter your image file: ");
    const des = readlineSync.question("Enter the description: ");

    currentImagePath = input;

    const embeddings = new CustomEmbeddings(getImageEmbedding);

    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
      collection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding",
    });

    // Create documents with descriptive text
    const documents = [
      {
        text: des,
        metadata: { imagePath: currentImagePath },
      },
    ];

    // Add documents to the vector store
    const result = await vectorStore.addDocuments(documents);

    console.log(
      `Imported ${result.length} documents into the MongoDB Atlas vector store.`
    );

    // Clean up MongoDB connection
    await client.close();
  } catch (error) {
    console.error("Main error:", error);
    await client.close();
  }
}

// Run the main function
main();

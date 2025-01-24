import express from "express";
import multer from "multer";
import { pipeline } from "@xenova/transformers";
import { Double, MongoClient } from "mongodb";
import * as fs from "fs";
import { configDotenv } from "dotenv";
configDotenv();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Configuration
const MONGODB_URI = process.env.MONGO_URI;
const DB_NAME = "bot";
const COLLECTION_NAME = "image";


app.set("view engine", "ejs");

app.use(express.static('public'));

app.use(express.json({ limit: "10mb" }));

app.use(express.urlencoded({ extended: true }));


// Initialize pipeline and MongoDB connection
let featureExtractor, client, db;

async function initializeApp() {
  try {
    // Initialize feature extractor
    featureExtractor = await pipeline(
      "image-feature-extraction",
      "Xenova/clip-vit-base-patch32"
    );

    // Connect to MongoDB
    client = new MongoClient(MONGODB_URI);
    await client.connect();
    db = client.db(DB_NAME);

    console.log("Application initialized successfully");
  } catch (error) {
    console.error("Initialization failed:", error);
    process.exit(1);
  }
}


app.get("/", (req, res) => {
  res.render("pages/index");
});


app.post("/search-image", upload.single("image"), async (req, res) => {
  let tempPath;
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }
    console.log(req.file)
    // 1. Save buffer to temp file
    tempPath = `./temp_${Date.now()}.jpg`;
    fs.writeFileSync(tempPath, req.file.buffer);

    // 2. Generate embedding
    const { data: embedding } = await featureExtractor(tempPath);

    console.log(embedding);
    // 3. Vector search in 
    
    const collection = db.collection(COLLECTION_NAME);
    const queryVector = Array.from(embedding);

 
    const cursor = await collection.aggregate([
      {
        $vectorSearch: {
          index: "image_index",
          path: "embedding",
          queryVector: queryVector,
          numCandidates: 100,
          limit: 3,
        },
      },
      {
        $project: {
          _id: 1,
          text: 1,
          imagePath: 1,
          score: { $meta: "vectorSearchScore" },
        },
      },
    ]);

    const results = await cursor.toArray();


    // 4. Clean up
    fs.unlinkSync(tempPath);

    res.json({
      queryImage: req.file.originalname,
      matches: results,
    });
  } catch (error) {
    if (tempPath && fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }
    console.error("Search failed:", error);
    res.status(500).json({ error: "Image search failed" });
  }
});

// Initialize and start server
initializeApp().then(() => {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});

// Cleanup on shutdown
process.on("SIGINT", async () => {
  await client.close();
  process.exit();
});

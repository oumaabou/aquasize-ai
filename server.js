import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'web')));

// Publishable (browser-safe) key — kept server-side here only to dodge Roboflow's
// CORS gap on detect.roboflow.com, not because it's a real secret.
const API_KEY = process.env.ROBOFLOW_API_KEY || 'rf_PyW41AdkpNTRRRGLIaZYKiwvDKp1';
const MODEL_ID = 'ouma-aboubaker-s-workspace/fish-identification-fw0uo-wyizm-1-yolo11n-t2';

// Proxies inference requests to Roboflow server-side, since detect.roboflow.com
// does not return Access-Control-Allow-Origin headers for browser-side fetches.
app.post('/api/infer', async (req, res) => {
  try {
    const confidence = req.body.confidence ?? 0.05;
    let url = `https://detect.roboflow.com/${MODEL_ID}?api_key=${API_KEY}&confidence=${confidence}`;
    const options = { method: 'POST' };

    if (req.body.imageUrl) {
      url += `&image=${encodeURIComponent(req.body.imageUrl)}`;
    } else if (req.body.image) {
      options.headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
      options.body = req.body.image;
    } else {
      return res.status(400).json({ error: 'Provide "image" (base64) or "imageUrl".' });
    }

    const rfRes = await fetch(url, options);
    const data = await rfRes.json();
    res.status(rfRes.status).json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AquaSize AI server listening on ${PORT}`));

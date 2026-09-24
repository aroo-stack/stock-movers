import express from 'express';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// Global error handlers (same pattern as insider-tracker)
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON = process.env.PYTHON || 'python3';
const SCRIPT = path.join(__dirname, 'stock_movers.py');

app.use(express.static('public'));

// Real symbols may contain '=', '^', '.', '-', '_'; a few spaces are allowed
// for multi-word sector names ("consumer discretionary"). Everything else
// (shell metacharacters, path separators, quotes, control chars) is rejected.
const TICKER_RE = /^[A-Za-z0-9 .\-_=^]{1,32}$/;

// Small in-memory result cache (caps repeated network hits within a session).
// Analysis pulls live data, so entries expire after a few minutes.
const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX = 12;
const analysisCache = new Map();

function friendlyError(stderr) {
  const tail = stderr.trim().split('\n').slice(-4).join('\n');
  if (!tail) return 'the analysis failed to finish (no error details).';
  return `the analysis failed: ${tail}`;
}

function runAnalysis(ticker) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const child = spawn(PYTHON, [SCRIPT, '--json', ticker], {
      cwd: __dirname,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });

    const killTimer = setTimeout(() => {
      console.error(`[${ticker}] timed out after 180s, killing analysis`);
      child.kill('SIGKILL');
    }, 180000);

    child.on('error', (err) => {
      clearTimeout(killTimer);
      reject(new Error(`could not start the analysis (${err.message}). ` +
                      `Make sure python3 is installed.`));
    });

    child.on('close', (code) => {
      clearTimeout(killTimer);
      const ms = Date.now() - t0;
      if (stderr.trim()) {
        console.log(`[${ticker}] analysis done (${ms}ms) -- python stderr:\n` +
                    stderr.trim().split('\n').slice(-6).join('\n'));
      }
      if (code !== 0) {
        reject(new Error(friendlyError(stderr)));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error('the analysis produced output the server could not ' +
                         'read; try again or run stock_movers.py directly.'));
      }
    });
  });
}

app.get('/api/analyze/:ticker', async (req, res) => {
  const raw = (req.params.ticker || '').trim();
  if (!TICKER_RE.test(raw)) {
    return res.status(400).json({
      error: `"${req.params.ticker}" doesn't look like a valid ticker or ` +
             `sector name. Try something like MU, TSLA, gold or technology.`,
    });
  }

  const cacheKey = raw.toUpperCase();
  const hit = analysisCache.get(cacheKey);
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
    console.log(`[${raw}] serving cached result (${Date.now() - hit.at}ms old)`);
    return res.json(hit.data);
  }
  if (hit) analysisCache.delete(cacheKey);

  try {
    const data = await runAnalysis(raw);
    analysisCache.set(cacheKey, { at: Date.now(), data });
    if (analysisCache.size > CACHE_MAX) {
      const oldest = [...analysisCache.entries()]
        .sort((a, b) => a[1].at - b[1].at)[0][0];
      analysisCache.delete(oldest);
    }
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Stock Movers running on http://localhost:${PORT}`);
  console.log(`Analysis engine: ${PYTHON} ${SCRIPT} --json <ticker>`);
});

export default app;
# Deploying Pattern Lens (free tier)

Page on Firebase Hosting, Python API on Cloud Run (Tokyo), forward log in Firestore, a scan every
15 minutes from Cloud Scheduler. At this size everything should stay inside Google's free monthly
quotas, but a billing account is required, so set a budget alert first (step 1).

## 0. Tools (once)

```bash
brew install --cask google-cloud-sdk
npm install -g firebase-tools
```

## 1. Project and billing safety

1. Create a project at console.firebase.google.com (this also creates the Google Cloud project).
2. Upgrade it to the Blaze plan (needed for Cloud Run, Firestore and Scheduler).
3. In Google Cloud console: Billing > Budgets & alerts > create budgets of USD 1 and USD 5 with email alerts.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com cloudscheduler.googleapis.com
gcloud firestore databases create --location=asia-northeast1
```

## 2. API on Cloud Run

Pick a long random token for the scanner, for example `openssl rand -hex 16`.

```bash
gcloud run deploy pattern-lens-api --source . --region asia-northeast1 \
  --allow-unauthenticated --max-instances 1 --memory 1Gi \
  --set-env-vars LOG_BACKEND=firestore,SCAN_TOKEN=YOUR_TOKEN
```

`--max-instances 1` caps cost. The command prints the service URL.

Test that Binance answers from Tokyo:

```bash
curl -X POST https://SERVICE_URL/log/scan -H "X-Scan-Token: YOUR_TOKEN"
```

If the reply shows `"added"` and the Cloud Run logs show no "fetch failed" lines, Binance is reachable.
If every fetch fails, Binance is blocking that region: skip step 4 and run the scanner on your own
computer instead (step 5). The page itself is not affected, because it sends its own candles to the API.

## 3. Page on Firebase Hosting

```bash
mkdir -p public && cp pattern-lens.html public/index.html
firebase login
firebase use YOUR_PROJECT_ID
firebase deploy --only hosting
```

The page detects that it is hosted and calls `/api/...`, which Firebase forwards to Cloud Run.

## 4. Scan every 15 minutes

```bash
gcloud scheduler jobs create http pattern-lens-scan --location asia-northeast1 \
  --schedule "*/15 * * * *" --uri https://SERVICE_URL/log/scan --http-method POST \
  --headers X-Scan-Token=YOUR_TOKEN
```

## 5. Fallback: scan from your own computer into the same Firestore

```bash
gcloud auth application-default login
LOG_BACKEND=firestore python forward_log.py --loop 900
```

## 6. AI analyst when hosted

Ollama cannot run on Cloud Run's free tier. Either leave the button as is (it says no engine is
reachable), or store an Anthropic API key in Secret Manager and pass it to the service:

```bash
printf "YOUR_KEY" | gcloud secrets create anthropic-key --data-file=-
gcloud run services update pattern-lens-api --region asia-northeast1 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

Set a monthly spending limit in the Anthropic console as well.

## Updating

- New models or research results: rerun step 2.
- Page changes: copy `pattern-lens.html` to `public/index.html` and rerun `firebase deploy --only hosting`.

## Notes

- Lightweight Charts is free but its licence asks for TradingView attribution; the page footer links to it.
- Not financial advice. Check the rules for publishing market analysis where you live before charging for it.

## Telegram alerts when hosted

Add the bot settings to the service (and ALERT_PATTERNS=1 for pattern events); the scheduled scan sends them:

```bash
gcloud run services update pattern-lens-api --region asia-northeast1 \
  --update-env-vars TELEGRAM_TOKEN=YOUR_BOT_TOKEN,TELEGRAM_CHAT_ID=YOUR_CHAT_ID,ALERT_PATTERNS=1
```

Alert state is kept in a local file, so on Cloud Run a restart can resend the latest signal once.

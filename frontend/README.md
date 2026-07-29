# AskingMe Agent Frontend

React and Vite user interface for the AskingMe Agent enterprise policy assistant.

## Local Development

```bash
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and calls the FastAPI backend at `http://127.0.0.1:8000` by default.

To override the backend address, copy `.env.example` to `.env.local` and change:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Validation

```bash
npm run lint
npm run build
```

The root Docker Compose configuration builds this frontend as an Nginx static site and accesses the backend through the same-origin `/api` path. The complete application is available at `http://localhost`.

/**
 * K6 Smoke Test — Quick sanity check (< 60s)
 * Run before the full load test to confirm API is healthy.
 *
 * k6 run tests/performance/smoke_test.js
 */

import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.API_URL || "http://localhost:8000";

export const options = {
  vus: 1,
  duration: "30s",
  thresholds: {
    http_req_duration: ["p(95)<500"],
    http_req_failed:   ["rate<0.01"],
  },
};

const PAYLOAD = JSON.stringify({
  age: 35,
  income: 75000,
  credit_score: 720,
  loan_amount: 25000,
  loan_term_months: 60,
  employment_type: "employed",
  marital_status: "married",
});

const HEADERS = { "Content-Type": "application/json" };

export default function () {
  // Health check
  const health = http.get(`${BASE_URL}/health`);
  check(health, { "health OK": (r) => r.status === 200 });

  // Single prediction
  const pred = http.post(`${BASE_URL}/v1/predict`, PAYLOAD, { headers: HEADERS });
  check(pred, {
    "predict status 200":  (r) => r.status === 200,
    "has prediction":      (r) => JSON.parse(r.body).prediction !== undefined,
    "prediction is binary":(r) => [0, 1].includes(JSON.parse(r.body).prediction),
  });

  sleep(1);
}

export function handleSummary(data) {
  const pass = data.metrics.http_req_failed.values.rate < 0.01;
  console.log(`\n🔬 Smoke Test: ${pass ? "✅ PASSED" : "❌ FAILED"}`);
  return {};
}

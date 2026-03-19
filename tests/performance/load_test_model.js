/**
 * K6 Performance Test — ML Prediction API
 * Tests: Load, Stress, Spike, and Soak scenarios.
 * Run: k6 run tests/performance/load_test_model.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Trend, Rate, Counter, Gauge } from 'k6/metrics';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// ── Custom Metrics ─────────────────────────────────────────────────────────
const inferenceLatency   = new Trend('inference_latency_ms', true);
const errorRate          = new Rate('error_rate');
const throughput         = new Counter('successful_predictions');
const confidenceScore    = new Gauge('avg_confidence_score');

// ── Thresholds (Quality Gate) ──────────────────────────────────────────────
export const options = {
  scenarios: {
    // 1. Baseline load test
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },   // Ramp up
        { duration: '3m', target: 50 },   // Sustained
        { duration: '1m', target: 0 },    // Ramp down
      ],
      tags: { scenario: 'load' },
    },

    // 2. Stress test (starts after load_test)
    stress_test: {
      executor: 'ramping-vus',
      startTime: '6m',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },  // Heavy load
        { duration: '3m', target: 200 },  // Breaking point
        { duration: '2m', target: 0 },
      ],
      tags: { scenario: 'stress' },
    },

    // 3. Spike test
    spike_test: {
      executor: 'ramping-vus',
      startTime: '13m',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 200 }, // Sudden spike
        { duration: '1m', target: 200 },  // Hold
        { duration: '10s', target: 0 },   // Drop
      ],
      tags: { scenario: 'spike' },
    },
  },

  thresholds: {
    // Latency SLOs
    'http_req_duration':                  ['p(50)<100', 'p(95)<300', 'p(99)<500'],
    'inference_latency_ms':               ['p(95)<300', 'p(99)<500', 'max<1000'],

    // Reliability SLOs
    'error_rate':                         ['rate<0.01'],      // < 1% errors
    'http_req_failed':                    ['rate<0.01'],

    // Throughput (only measured during load scenario)
    'http_req_duration{scenario:load}':   ['p(95)<250'],
    'http_req_duration{scenario:stress}': ['p(95)<500'],
    'http_req_duration{scenario:spike}':  ['p(95)<800'],
  },
};

// ── Config ─────────────────────────────────────────────────────────────────
const BASE_URL = __ENV.MODEL_API_URL || 'http://localhost:8000';
const PREDICT_URL = `${BASE_URL}/v1/predict`;
const HEALTH_URL  = `${BASE_URL}/health`;

const HEADERS = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${__ENV.MODEL_API_TOKEN || 'test-token'}`,
};

// ── Payload Generator ──────────────────────────────────────────────────────
function generatePayload() {
  const employmentTypes = ['employed', 'self_employed', 'unemployed', 'retired'];
  const maritalStatuses = ['single', 'married', 'divorced', 'widowed'];

  return JSON.stringify({
    age:               randomIntBetween(18, 75),
    income:            randomIntBetween(20000, 200000),
    credit_score:      randomIntBetween(300, 850),
    loan_amount:       randomIntBetween(1000, 100000),
    loan_term_months:  randomIntBetween(6, 120),
    employment_type:   employmentTypes[randomIntBetween(0, employmentTypes.length - 1)],
    marital_status:    maritalStatuses[randomIntBetween(0, maritalStatuses.length - 1)],
  });
}

// ── Setup (runs once before test) ─────────────────────────────────────────
export function setup() {
  const healthRes = http.get(HEALTH_URL, { headers: HEADERS });
  if (healthRes.status !== 200) {
    console.error(`❌ Health check failed! Status: ${healthRes.status}`);
  } else {
    console.log(`✅ API is healthy. Starting performance tests...`);
  }
  return { baseUrl: BASE_URL };
}

// ── Main VU function ───────────────────────────────────────────────────────
export default function () {
  group('Prediction API', () => {

    group('Valid Prediction', () => {
      const payload = generatePayload();
      const start   = Date.now();

      const res = http.post(PREDICT_URL, payload, {
        headers: HEADERS,
        tags: { name: 'predict_valid' },
      });

      const latency = Date.now() - start;
      inferenceLatency.add(latency);

      const ok = check(res, {
        '✅ status is 200':             (r) => r.status === 200,
        '✅ has prediction field':       (r) => {
          try { return JSON.parse(r.body).prediction !== undefined; }
          catch { return false; }
        },
        '✅ prediction is binary':       (r) => {
          try { return [0, 1].includes(JSON.parse(r.body).prediction); }
          catch { return false; }
        },
        '✅ confidence in [0,1]':        (r) => {
          try {
            const c = JSON.parse(r.body).confidence;
            return typeof c === 'number' && c >= 0 && c <= 1;
          } catch { return false; }
        },
        '✅ latency < 500ms':           () => latency < 500,
      });

      errorRate.add(!ok);
      if (ok) {
        throughput.add(1);
        try {
          const body = JSON.parse(res.body);
          if (body.confidence) confidenceScore.add(body.confidence);
        } catch {}
      }
    });

    group('Health Check', () => {
      const res = http.get(HEALTH_URL, { headers: HEADERS });
      check(res, {
        '✅ health status 200': (r) => r.status === 200,
      });
    });

  });

  sleep(0.5);
}

// ── Teardown (runs once after test) ───────────────────────────────────────
export function teardown(data) {
  console.log('\n📊 Performance Test Complete');
  console.log(`Base URL: ${data.baseUrl}`);
}

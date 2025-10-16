import axios from "axios";

const API_ROOT = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const fetchAuthors = () => axios.get(`${API_ROOT}/authors`).then(r => r.data);
export const fetchGraph = (top_k=100) => axios.get(`${API_ROOT}/graph?top_k=${top_k}`).then(r=>r.data);
export const predictPair = (a,b) => axios.post(`${API_ROOT}/predict`, {author_a: a, author_b: b}).then(r=>r.data);
export const trainModel = () => axios.post(`${API_ROOT}/train`).then(r=>r.data);

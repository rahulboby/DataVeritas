import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 120000, // 2min for large datasets
});

// Dataset APIs
export const uploadDataset = (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
};

export const generateDataset = (nRows = 20000) =>
    api.post(`/generate?n_rows=${nRows}`);

export const getPreview = (datasetId, limit = 100) =>
    api.get(`/preview?dataset_id=${datasetId}&limit=${limit}`);

export const listDatasets = () => api.get('/datasets');

export const deleteDataset = (datasetId) =>
    api.delete(`/dataset/${datasetId}`);

// Analytics APIs
export const computeScore = (datasetId) =>
    api.post('/score', { dataset_id: datasetId });

export const computeFieldScore = (datasetId, column) =>
    api.post('/score/field', { dataset_id: datasetId, column });

export const computeDynamicScore = (datasetId) =>
    api.post('/score/dynamic', { dataset_id: datasetId });

export const predictQuality = (datasetId) =>
    api.post('/score/predict', { dataset_id: datasetId });

export const computeNulls = (datasetId) =>
    api.post('/nulls', { dataset_id: datasetId });

export const computeDuplicates = (datasetId) =>
    api.post('/duplicates', { dataset_id: datasetId });

export const computeEntityGraph = (datasetId, keyColumns, threshold) =>
    api.post('/duplicates/entity-graph', {
        dataset_id: datasetId,
        key_columns: keyColumns,
        similarity_threshold: threshold,
    });

export const computeOutliers = (datasetId) =>
    api.post('/outliers', { dataset_id: datasetId });

export const computeConsistency = (datasetId, rules = null) =>
    api.post('/consistency', { dataset_id: datasetId, rules });

export const discoverRules = (datasetId) =>
    api.post('/consistency/discover-rules', { dataset_id: datasetId });

export const computeDistribution = (datasetId) =>
    api.post('/distribution', { dataset_id: datasetId });

export const computeProfile = (datasetId) =>
    api.post('/distribution/profile', { dataset_id: datasetId });

export const generateInsights = (datasetId, rules = null) =>
    api.post('/insights', { dataset_id: datasetId, rules });

export default api;

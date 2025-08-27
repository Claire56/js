

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import heapq
from datetime import datetime, timedelta
import json
import logging

# ============================================================================
# QUESTION 1: User Behavior Analysis & Feature Engineering
# ============================================================================

class UserBehaviorAnalyzer:
    """
    Analyze user behavior patterns from Jira/Confluence logs
    Extract features for ML models
    """
    
    def __init__(self):
        self.user_sessions = defaultdict(list)
        self.feature_usage = defaultdict(int)
    
    def process_user_logs(self, log_entries: List[Dict]) -> Dict:
        """
        Process raw log entries to extract user behavior features
        
        Args:
            log_entries: List of log dictionaries with user_id, action, timestamp, etc.
            
        Returns:
            Dictionary with user behavior features
        """
        user_features = defaultdict(lambda: {
            'total_actions': 0,
            'unique_features': set(),
            'session_count': 0,
            'avg_session_duration': 0,
            'peak_hours': defaultdict(int),
            'error_rate': 0,
            'last_active': None
        })
        
        for entry in log_entries:
            user_id = entry['user_id']
            action = entry['action']
            timestamp = datetime.fromisoformat(entry['timestamp'])
            hour = timestamp.hour
            
            # Update basic metrics
            user_features[user_id]['total_actions'] += 1
            user_features[user_id]['unique_features'].add(action)
            user_features[user_id]['peak_hours'][hour] += 1
            user_features[user_id]['last_active'] = timestamp
            
            # Track errors
            if entry.get('status') == 'error':
                user_features[user_id]['error_rate'] += 1
        
        # Calculate derived features
        for user_id, features in user_features.items():
            features['unique_features'] = len(features['unique_features'])
            features['error_rate'] /= features['total_actions']
            features['peak_hour'] = max(features['peak_hours'].items(), key=lambda x: x[1])[0]
            
        return dict(user_features)
    
    def build_ml_features(self, user_data: Dict) -> pd.DataFrame:
        """
        Transform user behavior data into ML-ready features
        """
        features = []
        
        for user_id, data in user_data.items():
            feature_vector = {
                'user_id': user_id,
                'total_actions': data['total_actions'],
                'unique_features': data['unique_features'],
                'error_rate': data['error_rate'],
                'peak_hour': data['peak_hour'],
                'days_since_active': (datetime.now() - data['last_active']).days if data['last_active'] else 999
            }
            features.append(feature_vector)
        
        return pd.DataFrame(features)

# ============================================================================
# QUESTION 2: Recommendation System Implementation
# ============================================================================

class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommendation system for Atlassian products
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarities = {}
        self.item_similarities = {}
    
    def build_user_item_matrix(self, user_actions: List[Dict]) -> np.ndarray:
        """
        Build user-item interaction matrix
        
        Args:
            user_actions: List of user action dictionaries
            
        Returns:
            User-item matrix where matrix[i][j] = interaction strength
        """
        # Create user and item mappings
        users = list(set(action['user_id'] for action in user_actions))
        items = list(set(action['item_id'] for action in user_actions))
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        # Build matrix
        matrix = np.zeros((len(users), len(items)))
        
        for action in user_actions:
            user_idx = user_to_idx[action['user_id']]
            item_idx = item_to_idx[action['item_id']]
            # Weight by action type (view=1, like=2, comment=3, etc.)
            weight = self._get_action_weight(action['action_type'])
            matrix[user_idx][item_idx] += weight
        
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.user_item_matrix = matrix
        
        return matrix
    
    def _get_action_weight(self, action_type: str) -> float:
        """Convert action type to weight"""
        weights = {
            'view': 1.0,
            'like': 2.0,
            'comment': 3.0,
            'share': 4.0,
            'bookmark': 5.0
        }
        return weights.get(action_type, 1.0)
    
    def calculate_user_similarities(self) -> Dict:
        """Calculate cosine similarity between all users"""
        if self.user_item_matrix is None:
            raise ValueError("Must build user-item matrix first")
        
        n_users = self.user_item_matrix.shape[0]
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity = self._cosine_similarity(
                    self.user_item_matrix[i], 
                    self.user_item_matrix[j]
                )
                self.user_similarities[(i, j)] = similarity
                self.user_similarities[(j, i)] = similarity
        
        return self.user_similarities
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[user_idx]
        
        # Find similar users
        similar_users = []
        for other_user_idx, similarity in self.user_similarities.items():
            if other_user_idx[0] == user_idx and similarity > 0:
                similar_users.append((other_user_idx[1], similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate recommendation scores
        item_scores = defaultdict(float)
        user_items = set(np.where(user_vector > 0)[0])  # Items user has already interacted with
        
        for similar_user_idx, similarity in similar_users[:10]:  # Top 10 similar users
            similar_user_vector = self.user_item_matrix[similar_user_idx]
            
            for item_idx, interaction_strength in enumerate(similar_user_vector):
                if interaction_strength > 0 and item_idx not in user_items:
                    item_scores[item_idx] += similarity * interaction_strength
        
        # Sort and return top recommendations
        recommendations = [(self.idx_to_item[item_idx], score) 
                          for item_idx, score in item_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]

# ============================================================================
# QUESTION 3: Anomaly Detection System
# ============================================================================

class AnomalyDetector:
    """
    Detect anomalous user behavior and system performance issues
    """
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.user_baselines = {}
        self.system_metrics = []
    
    def detect_user_anomalies(self, user_actions: List[Dict]) -> List[Dict]:
        """
        Detect unusual user behavior patterns
        
        Args:
            user_actions: List of recent user actions
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Group actions by user
        user_action_counts = defaultdict(int)
        user_action_types = defaultdict(set)
        user_timestamps = defaultdict(list)
        
        for action in user_actions:
            user_id = action['user_id']
            user_action_counts[user_id] += 1
            user_action_types[user_id].add(action['action_type'])
            user_timestamps[user_id].append(datetime.fromisoformat(action['timestamp']))
        
        # Check for anomalies
        for user_id, action_count in user_action_counts.items():
            baseline = self.user_baselines.get(user_id, {'avg_actions': 10, 'std_actions': 5})
            
            # Check for unusual activity volume
            if abs(action_count - baseline['avg_actions']) > self.threshold * baseline['std_actions']:
                anomalies.append({
                    'user_id': user_id,
                    'anomaly_type': 'unusual_activity_volume',
                    'severity': 'high' if action_count > baseline['avg_actions'] * 3 else 'medium',
                    'details': f'User performed {action_count} actions vs baseline {baseline["avg_actions"]}'
                })
            
            # Check for unusual action types
            if len(user_action_types[user_id]) > 10:  # Too many different action types
                anomalies.append({
                    'user_id': user_id,
                    'anomaly_type': 'unusual_action_diversity',
                    'severity': 'medium',
                    'details': f'User performed {len(user_action_types[user_id])} different action types'
                })
            
            # Check for rapid successive actions (potential automation)
            if len(user_timestamps[user_id]) > 1:
                time_diffs = []
                sorted_times = sorted(user_timestamps[user_id])
                for i in range(1, len(sorted_times)):
                    time_diffs.append((sorted_times[i] - sorted_times[i-1]).total_seconds())
                
                if any(diff < 1.0 for diff in time_diffs):  # Actions less than 1 second apart
                    anomalies.append({
                        'user_id': user_id,
                        'anomaly_type': 'rapid_successive_actions',
                        'severity': 'high',
                        'details': 'User performed actions with <1 second intervals'
                    })
        
        return anomalies
    
    def detect_performance_anomalies(self, metrics: List[Dict]) -> List[Dict]:
        """
        Detect system performance anomalies
        
        Args:
            metrics: List of system metrics dictionaries
            
        Returns:
            List of detected performance anomalies
        """
        anomalies = []
        
        # Extract key metrics
        response_times = [m['response_time'] for m in metrics]
        error_rates = [m['error_rate'] for m in metrics]
        cpu_usage = [m['cpu_usage'] for m in metrics]
        
        # Calculate baseline statistics
        if len(response_times) > 10:
            rt_mean = np.mean(response_times)
            rt_std = np.std(response_times)
            
            # Check for response time spikes
            for i, rt in enumerate(response_times):
                if rt > rt_mean + 2 * rt_std:
                    anomalies.append({
                        'metric': 'response_time',
                        'timestamp': metrics[i]['timestamp'],
                        'value': rt,
                        'threshold': rt_mean + 2 * rt_std,
                        'severity': 'high' if rt > rt_mean + 3 * rt_std else 'medium'
                    })
        
        # Check error rate spikes
        if len(error_rates) > 5:
            error_mean = np.mean(error_rates)
            for i, er in enumerate(error_rates):
                if er > error_mean * 2:  # Error rate doubled
                    anomalies.append({
                        'metric': 'error_rate',
                        'timestamp': metrics[i]['timestamp'],
                        'value': er,
                        'threshold': error_mean * 2,
                        'severity': 'high'
                    })
        
        return anomalies

# ============================================================================
# QUESTION 4: A/B Testing Framework
# ============================================================================

class ABTestFramework:
    """
    A/B testing framework for ML model evaluation
    """
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(self, experiment_id: str, control_group: float, 
                         treatment_group: float, metrics: List[str]) -> Dict:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            control_group: Percentage of users in control group (0-1)
            treatment_group: Percentage of users in treatment group (0-1)
            metrics: List of metrics to track
            
        Returns:
            Experiment configuration
        """
        if control_group + treatment_group > 1.0:
            raise ValueError("Control + treatment groups cannot exceed 100%")
        
        experiment = {
            'experiment_id': experiment_id,
            'control_group': control_group,
            'treatment_group': treatment_group,
            'metrics': metrics,
            'start_time': datetime.now(),
            'status': 'active',
            'control_data': defaultdict(list),
            'treatment_data': defaultdict(list)
        }
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def assign_user_to_group(self, user_id: str, experiment_id: str) -> str:
        """
        Assign user to control or treatment group
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            
        Returns:
            'control' or 'treatment'
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Use hash of user_id for consistent assignment
        hash_value = hash(user_id + experiment_id) % 100
        
        experiment = self.experiments[experiment_id]
        control_threshold = experiment['control_group'] * 100
        
        if hash_value < control_threshold:
            return 'control'
        else:
            return 'treatment'
    
    def record_metric(self, experiment_id: str, user_id: str, 
                     metric_name: str, value: float) -> None:
        """
        Record a metric value for a user in an experiment
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            metric_name: Name of the metric
            value: Metric value
        """
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        group = self.assign_user_to_group(user_id, experiment_id)
        
        if group == 'control':
            experiment['control_data'][metric_name].append(value)
        else:
            experiment['treatment_data'][metric_name].append(value)
    
    def analyze_results(self, experiment_id: str, confidence_level: float = 0.95) -> Dict:
        """
        Analyze A/B test results with statistical significance
        
        Args:
            experiment_id: Experiment identifier
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Analysis results with p-values and confidence intervals
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        results = {}
        
        for metric in experiment['metrics']:
            control_data = experiment['control_data'][metric]
            treatment_data = experiment['treatment_data'][metric]
            
            if len(control_data) == 0 or len(treatment_data) == 0:
                continue
            
            # Calculate basic statistics
            control_mean = np.mean(control_data)
            treatment_mean = np.mean(treatment_data)
            control_std = np.std(control_data, ddof=1)
            treatment_std = np.std(treatment_data, ddof=1)
            
            # Calculate t-statistic and p-value
            n1, n2 = len(control_data), len(treatment_data)
            
            # Pooled standard error
            pooled_se = np.sqrt((control_std**2 / n1) + (treatment_std**2 / n2))
            
            # t-statistic
            t_stat = (treatment_mean - control_mean) / pooled_se
            
            # Degrees of freedom (simplified)
            df = min(n1, n2) - 1
            
            # Calculate p-value (simplified - in practice use scipy.stats)
            p_value = self._calculate_p_value(t_stat, df)
            
            # Calculate confidence interval
            margin_of_error = 1.96 * pooled_se  # 95% confidence
            ci_lower = (treatment_mean - control_mean) - margin_of_error
            ci_upper = (treatment_mean - control_mean) + margin_of_error
            
            # Determine significance
            is_significant = p_value < (1 - confidence_level)
            
            results[metric] = {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'difference': treatment_mean - control_mean,
                'p_value': p_value,
                'is_significant': is_significant,
                'confidence_interval': (ci_lower, ci_upper),
                'effect_size': (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
            }
        
        self.results[experiment_id] = results
        return results
    
    def _calculate_p_value(self, t_stat: float, df: int) -> float:
        """Simplified p-value calculation (use scipy.stats.t.sf in practice)"""
        # This is a simplified approximation
        if abs(t_stat) > 3:
            return 0.001
        elif abs(t_stat) > 2:
            return 0.05
        elif abs(t_stat) > 1.5:
            return 0.1
        else:
            return 0.5

# ============================================================================
# QUESTION 5: Real-time Stream Processing
# ============================================================================

class StreamProcessor:
    """
    Real-time processing of user events for ML features
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.event_buffer = deque(maxlen=window_size)
        self.user_sessions = defaultdict(list)
        self.real_time_features = defaultdict(dict)
    
    def process_event(self, event: Dict) -> Dict:
        """
        Process a single user event in real-time
        
        Args:
            event: User event dictionary
            
        Returns:
            Real-time features for the user
        """
        user_id = event['user_id']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Update user session
        self.user_sessions[user_id].append({
            'action': event['action'],
            'timestamp': timestamp,
            'context': event.get('context', {})
        })
        
        # Calculate real-time features
        features = self._calculate_realtime_features(user_id, timestamp)
        self.real_time_features[user_id] = features
        
        return features
    
    def _calculate_realtime_features(self, user_id: str, current_time: datetime) -> Dict:
        """
        Calculate real-time features for a user
        """
        user_events = self.user_sessions[user_id]
        
        # Filter events from last hour
        one_hour_ago = current_time - timedelta(hours=1)
        recent_events = [e for e in user_events if e['timestamp'] > one_hour_ago]
        
        features = {
            'user_id': user_id,
            'events_last_hour': len(recent_events),
            'unique_actions_last_hour': len(set(e['action'] for e in recent_events)),
            'session_duration': self._calculate_session_duration(user_events),
            'action_frequency': len(recent_events) / 60.0,  # events per minute
            'last_action': user_events[-1]['action'] if user_events else None,
            'time_since_last_action': (current_time - user_events[-1]['timestamp']).total_seconds() if user_events else 999999
        }
        
        return features
    
    def _calculate_session_duration(self, user_events: List[Dict]) -> float:
        """Calculate current session duration in minutes"""
        if len(user_events) < 2:
            return 0.0
        
        # Group events into sessions (gap > 30 minutes = new session)
        sessions = []
        current_session = [user_events[0]]
        
        for i in range(1, len(user_events)):
            time_diff = (user_events[i]['timestamp'] - user_events[i-1]['timestamp']).total_seconds()
            
            if time_diff > 1800:  # 30 minutes
                sessions.append(current_session)
                current_session = [user_events[i]]
            else:
                current_session.append(user_events[i])
        
        sessions.append(current_session)
        
        # Return duration of current session
        current_session = sessions[-1]
        if len(current_session) > 1:
            duration = (current_session[-1]['timestamp'] - current_session[0]['timestamp']).total_seconds()
            return duration / 60.0  # Convert to minutes
        
        return 0.0
    
    def get_realtime_recommendations(self, user_id: str) -> List[Dict]:
        """
        Generate real-time recommendations based on current user state
        """
        if user_id not in self.real_time_features:
            return []
        
        features = self.real_time_features[user_id]
        recommendations = []
        
        # Simple rule-based recommendations
        if features['events_last_hour'] > 20:
            recommendations.append({
                'type': 'high_activity',
                'message': 'You\'re very active! Consider taking a break.',
                'priority': 'low'
            })
        
        if features['time_since_last_action'] > 3600:  # 1 hour
            recommendations.append({
                'type': 're_engagement',
                'message': 'Welcome back! Here are some recent updates.',
                'priority': 'high'
            })
        
        if features['unique_actions_last_hour'] < 3:
            recommendations.append({
                'type': 'feature_discovery',
                'message': 'Try exploring new features to boost productivity!',
                'priority': 'medium'
            })
        
        return recommendations

# ============================================================================
# QUESTION 6: Model Serving & Performance Optimization
# ============================================================================

class ModelServingSystem:
    """
    Optimized model serving system for production ML
    """
    
    def __init__(self, model_cache_size: int = 100):
        self.model_cache = {}
        self.model_cache_size = model_cache_size
        self.request_queue = deque()
        self.batch_size = 32
        self.performance_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'average_latency': 0.0,
            'error_count': 0
        }
    
    def load_model(self, model_id: str, model_path: str) -> bool:
        """
        Load a model into the serving system
        
        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Simulate model loading (in practice, load actual model)
            model = {
                'id': model_id,
                'path': model_path,
                'loaded_at': datetime.now(),
                'predictions_made': 0
            }
            
            # Add to cache (LRU eviction if needed)
            if len(self.model_cache) >= self.model_cache_size:
                # Remove oldest model
                oldest_model = min(self.model_cache.values(), key=lambda x: x['loaded_at'])
                del self.model_cache[oldest_model['id']]
            
            self.model_cache[model_id] = model
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def predict(self, model_id: str, input_data: List[Dict]) -> List[Dict]:
        """
        Make predictions using the specified model
        
        Args:
            model_id: Model identifier
            input_data: List of input data dictionaries
            
        Returns:
            List of prediction results
        """
        start_time = datetime.now()
        
        try:
            # Check if model is loaded
            if model_id not in self.model_cache:
                if not self.load_model(model_id, f"models/{model_id}.pkl"):
                    raise ValueError(f"Model {model_id} not found")
            
            # Simulate model prediction
            predictions = []
            for data in input_data:
                # Simulate prediction logic
                prediction = {
                    'user_id': data.get('user_id'),
                    'prediction': self._simulate_prediction(data),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'timestamp': datetime.now().isoformat()
                }
                predictions.append(prediction)
            
            # Update metrics
            latency = (datetime.now() - start_time).total_seconds()
            self._update_metrics(latency, cache_hit=True)
            
            # Update model usage
            self.model_cache[model_id]['predictions_made'] += len(predictions)
            
            return predictions
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            logging.error(f"Prediction error: {e}")
            raise
    
    def _simulate_prediction(self, data: Dict) -> str:
        """Simulate model prediction (replace with actual model inference)"""
        # Simple simulation based on input data
        if data.get('user_activity', 0) > 50:
            return 'high_engagement'
        elif data.get('user_activity', 0) > 20:
            return 'medium_engagement'
        else:
            return 'low_engagement'
    
    def _update_metrics(self, latency: float, cache_hit: bool = False):
        """Update performance metrics"""
        self.performance_metrics['total_requests'] += 1
        if cache_hit:
            self.performance_metrics['cache_hits'] += 1
        
        # Update average latency
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['average_latency']
        self.performance_metrics['average_latency'] = (
            (current_avg * (total_requests - 1) + latency) / total_requests
        )
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        metrics['cache_hit_rate'] = (
            metrics['cache_hits'] / metrics['total_requests'] 
            if metrics['total_requests'] > 0 else 0
        )
        metrics['error_rate'] = (
            metrics['error_count'] / metrics['total_requests']
            if metrics['total_requests'] > 0 else 0
        )
        return metrics
    
    def batch_predict(self, model_id: str, input_batch: List[Dict]) -> List[Dict]:
        """
        Process predictions in batches for better performance
        
        Args:
            model_id: Model identifier
            input_batch: Batch of input data
            
        Returns:
            Batch of predictions
        """
        if len(input_batch) > self.batch_size:
            # Split into smaller batches
            results = []
            for i in range(0, len(input_batch), self.batch_size):
                batch = input_batch[i:i + self.batch_size]
                batch_results = self.predict(model_id, batch)
                results.extend(batch_results)
            return results
        else:
            return self.predict(model_id, input_batch)

# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================

def run_example_usage():
    """Demonstrate usage of all components"""
    print("=== Atlassian ML Systems Engineer Coding Examples ===\n")
    
    # Example 1: User Behavior Analysis
    print("1. User Behavior Analysis")
    analyzer = UserBehaviorAnalyzer()
    
    # Sample log data
    log_entries = [
        {'user_id': 'user1', 'action': 'view_page', 'timestamp': '2024-01-01T10:00:00', 'status': 'success'},
        {'user_id': 'user1', 'action': 'create_issue', 'timestamp': '2024-01-01T10:05:00', 'status': 'success'},
        {'user_id': 'user2', 'action': 'view_page', 'timestamp': '2024-01-01T10:10:00', 'status': 'error'},
    ]
    
    user_features = analyzer.process_user_logs(log_entries)
    print(f"Processed {len(user_features)} users")
    print(f"Sample features: {list(user_features.values())[0]}\n")
    
    # Example 2: Recommendation System
    print("2. Recommendation System")
    recommender = CollaborativeFilteringRecommender()
    
    # Sample user actions
    user_actions = [
        {'user_id': 'user1', 'item_id': 'project1', 'action_type': 'view'},
        {'user_id': 'user1', 'item_id': 'project2', 'action_type': 'like'},
        {'user_id': 'user2', 'item_id': 'project1', 'action_type': 'view'},
    ]
    
    matrix = recommender.build_user_item_matrix(user_actions)
    print(f"Built user-item matrix: {matrix.shape}")
    
    recommender.calculate_user_similarities()
    recommendations = recommender.get_recommendations('user1', 5)
    print(f"Recommendations for user1: {recommendations}\n")
    
    # Example 3: Anomaly Detection
    print("3. Anomaly Detection")
    detector = AnomalyDetector()
    
    # Sample user actions for anomaly detection
    user_actions = [
        {'user_id': 'user1', 'action_type': 'view', 'timestamp': '2024-01-01T10:00:00'},
        {'user_id': 'user1', 'action_type': 'view', 'timestamp': '2024-01-01T10:00:01'},  # Suspicious timing
    ]
    
    anomalies = detector.detect_user_anomalies(user_actions)
    print(f"Detected {len(anomalies)} anomalies")
    for anomaly in anomalies:
        print(f"  - {anomaly['anomaly_type']}: {anomaly['details']}\n")
    
    # Example 4: A/B Testing
    print("4. A/B Testing Framework")
    ab_test = ABTestFramework()
    
    # Create experiment
    experiment = ab_test.create_experiment(
        'recommendation_test', 
        control_group=0.5, 
        treatment_group=0.5, 
        metrics=['click_rate', 'conversion_rate']
    )
    print(f"Created experiment: {experiment['experiment_id']}")
    
    # Simulate some data
    for i in range(100):
        user_id = f"user{i}"
        group = ab_test.assign_user_to_group(user_id, 'recommendation_test')
        ab_test.record_metric('recommendation_test', user_id, 'click_rate', 
                             np.random.normal(0.1 if group == 'control' else 0.15, 0.05))
    
    results = ab_test.analyze_results('recommendation_test')
    print(f"Test results: {results}\n")
    
    # Example 5: Stream Processing
    print("5. Real-time Stream Processing")
    processor = StreamProcessor()
    
    # Process some events
    events = [
        {'user_id': 'user1', 'action': 'view_page', 'timestamp': '2024-01-01T10:00:00'},
        {'user_id': 'user1', 'action': 'create_issue', 'timestamp': '2024-01-01T10:05:00'},
    ]
    
    for event in events:
        features = processor.process_event(event)
        print(f"Real-time features for {event['user_id']}: {features}")
    
    recommendations = processor.get_realtime_recommendations('user1')
    print(f"Real-time recommendations: {recommendations}\n")
    
    # Example 6: Model Serving
    print("6. Model Serving System")
    serving_system = ModelServingSystem()
    
    # Load model
    success = serving_system.load_model('recommendation_model', 'models/recommendation.pkl')
    print(f"Model loaded: {success}")
    
    # Make predictions
    input_data = [
        {'user_id': 'user1', 'user_activity': 30},
        {'user_id': 'user2', 'user_activity': 60},
    ]
    
    predictions = serving_system.predict('recommendation_model', input_data)
    print(f"Predictions: {predictions}")
    
    metrics = serving_system.get_performance_metrics()
    print(f"Performance metrics: {metrics}")

if __name__ == "__main__":
    run_example_usage()

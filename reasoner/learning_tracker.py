"""Learning Tracker - מעקב אחרי כל נוירון בנפרד ושימוש בצעדי הלמידה כטריגרים."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict
from .config import ModelConfig, default_config


class NeuronTracker:
    """עוקב אחרי נוירון אחד - איך הוא משתנה ומתפתח."""
    
    def __init__(self, neuron_id: str, config: ModelConfig = default_config):
        self.neuron_id = neuron_id
        self.trajectory: deque = deque(maxlen=config.learning_tracker_window)
        self.status = "learning"  # learning, strengthening, struggling, converged
        self.learning_steps = []  # הצעדים שבהם הוא למד
        self.config = config
    
    def track_step(self, step: int, weight_before: float, gradient: float, 
                   weight_after: float, loss: float) -> None:
        """שומר צעד אחד של למידה של הנוירון."""
        weight_change = weight_after - weight_before
        
        step_data = {
            'step': step,
            'weight_before': weight_before,
            'gradient': gradient,
            'weight_after': weight_after,
            'weight_change': weight_change,
            'loss': loss,
            'gradient_magnitude': abs(gradient),
            'weight_change_magnitude': abs(weight_change),
        }
        
        self.trajectory.append(step_data)
        self._update_status()
        
        # אם הנוירון למד משהו משמעותי - שמור את הצעד
        if abs(weight_change) > 1e-6 or abs(gradient) > 1e-6:
            self.learning_steps.append(step_data)
    
    def _update_status(self) -> None:
        """מעדכן את הסטטוס של הנוירון לפי המסלול שלו."""
        if len(self.trajectory) < 3:
            self.status = "learning"
            return
        
        recent = list(self.trajectory)
        gradients = [s['gradient_magnitude'] for s in recent]
        weight_changes = [s['weight_change_magnitude'] for s in recent]
        
        # בודק אם מתחזק (gradients קטנים יותר)
        if len(gradients) >= 3:
            if gradients[-1] < gradients[0] * self.config.gradient_strengthening_threshold:
                self.status = "strengthening"
                return
        
        # בודק אם נאבק (gradients גדלים)
        if len(gradients) >= 3:
            if gradients[-1] > gradients[0] * self.config.gradient_struggling_threshold:
                self.status = "struggling"
                return
        
        # בודק אם התכנס (שינויים קטנים מאוד)
        if len(weight_changes) >= self.config.convergence_window:
            recent_changes = weight_changes[-self.config.convergence_window:]
            if max(recent_changes) < self.config.convergence_threshold:
                self.status = "converged"
                return
        
        self.status = "learning"
    
    def get_triggers(self) -> Dict[str, Any]:
        """מחזיר טריגרים מהצעדים שבהם הנוירון למד."""
        triggers = {
            'should_accelerate': False,
            'should_slow_down': False,
            'should_freeze': False,
            'learning_strength': 0.0,
        }
        
        if not self.learning_steps:
            return triggers
        
        # מחשב את חוזק הלמידה לפי הצעדים האחרונים
        recent_steps = self.learning_steps[-5:] if len(self.learning_steps) >= 5 else self.learning_steps
        avg_change = np.mean([abs(s['weight_change']) for s in recent_steps])
        avg_gradient = np.mean([abs(s['gradient']) for s in recent_steps])
        
        triggers['learning_strength'] = float(avg_change + avg_gradient)
        
        # טריגרים לפי הסטטוס
        if self.status == "strengthening":
            triggers['should_accelerate'] = True
        elif self.status == "struggling":
            triggers['should_slow_down'] = True
        elif self.status == "converged":
            triggers['should_freeze'] = True
        
        return triggers


class LearningTracker:
    """עוקב אחרי כל נוירון בנפרד ומשתמש בצעדי הלמידה כטריגרים."""
    
    def __init__(self, window_size: Optional[int] = None, config: ModelConfig = default_config):
        """
        Args:
            window_size: כמה צעדים אחרונים לשמור לכל נוירון
            config: קונפיגורציה
        """
        self.config = config
        self.window_size = window_size if window_size is not None else config.learning_tracker_window
        self.neuron_trackers: Dict[str, NeuronTracker] = {}  # neuron_id -> tracker
        self.global_loss_history: deque = deque(maxlen=self.window_size)
    
    def track_neuron_step(self, neuron_id: str, step: int, weight_before: float,
                         gradient: float, weight_after: float, loss: float) -> None:
        """עוקב אחרי צעד אחד של נוירון אחד."""
        if neuron_id not in self.neuron_trackers:
            self.neuron_trackers[neuron_id] = NeuronTracker(neuron_id, config=self.config)
        
        tracker = self.neuron_trackers[neuron_id]
        tracker.track_step(step, weight_before, gradient, weight_after, loss)
        
        # שומר גם loss גלובלי
        if len(self.global_loss_history) == 0 or self.global_loss_history[-1]['step'] != step:
            self.global_loss_history.append({'step': step, 'loss': loss})
    
    def get_neuron_triggers(self, neuron_id: str) -> Dict[str, Any]:
        """מחזיר טריגרים של נוירון ספציפי."""
        if neuron_id not in self.neuron_trackers:
            return {}
        return self.neuron_trackers[neuron_id].get_triggers()
    
    def get_all_triggers(self) -> Dict[str, Dict[str, Any]]:
        """מחזיר טריגרים של כל הנוירונים."""
        return {nid: tracker.get_triggers() for nid, tracker in self.neuron_trackers.items()}
    
    def get_adaptive_learning_rate(self, neuron_id: str, base_lr: float) -> float:
        """מחזיר learning rate מותאם לנוירון ספציפי לפי הטריגרים שלו."""
        triggers = self.get_neuron_triggers(neuron_id)
        
        if not triggers:
            return base_lr
        
        lr = base_lr
        
        if triggers.get('should_accelerate'):
            # הנוירון מתחזק - אפשר להאיץ
            lr *= self.config.learning_rate_accelerate
        elif triggers.get('should_slow_down'):
            # הנוירון נאבק - צריך להאט
            lr *= self.config.learning_rate_slow_down
        elif triggers.get('should_freeze'):
            # הנוירון התכנס - אפשר לקפוא
            lr *= self.config.learning_rate_freeze
        
        # מגביל את ה-learning rate
        return np.clip(lr, self.config.learning_rate_min, self.config.learning_rate_max)
    
    def get_learning_signals(self) -> Dict[str, Any]:
        """מחזיר סיגנלים גלובליים מהלמידה."""
        signals = {
            'total_neurons': len(self.neuron_trackers),
            'strengthening_neurons': 0,
            'struggling_neurons': 0,
            'converged_neurons': 0,
            'learning_neurons': 0,
            'avg_learning_strength': 0.0,
        }
        
        if not self.neuron_trackers:
            return signals
        
        learning_strengths = []
        for tracker in self.neuron_trackers.values():
            if tracker.status == "strengthening":
                signals['strengthening_neurons'] += 1
            elif tracker.status == "struggling":
                signals['struggling_neurons'] += 1
            elif tracker.status == "converged":
                signals['converged_neurons'] += 1
            else:
                signals['learning_neurons'] += 1
            
            triggers = tracker.get_triggers()
            learning_strengths.append(triggers.get('learning_strength', 0.0))
        
        signals['avg_learning_strength'] = float(np.mean(learning_strengths)) if learning_strengths else 0.0
        
        return signals

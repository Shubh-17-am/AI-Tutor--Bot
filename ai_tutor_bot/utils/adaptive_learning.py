from datetime import datetime, timedelta
from typing import List, Dict
from ai_tutor_bot.utils.config import Config
import logging

logger = logging.getLogger(__name__)

class AdaptiveLearningSystem:
    def __init__(self):
        self.user_progress = {}  # type: Dict[str, Dict[str, List[datetime]]]

    def update_progress(self, user_id: str, concept: str) -> datetime:
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        if concept not in self.user_progress[user_id]:
            self.user_progress[user_id][concept] = []
        
        self.user_progress[user_id][concept].append(datetime.now())
        
        reviews = self.user_progress[user_id][concept]
        if len(reviews) < len(Config.REPETITION_INTERVALS):
            next_review = reviews[-1] + timedelta(days=Config.REPETITION_INTERVALS[len(reviews)-1])
        else:
            next_review = reviews[-1] + timedelta(days=Config.REPETITION_INTERVALS[-1])
        
        return next_review

    def get_learning_context(self, user_id: str, concepts: List[str]) -> List[str]:
        prioritized = []
        for concept in concepts:
            if user_id in self.user_progress and concept in self.user_progress[user_id]:
                reviews = self.user_progress[user_id][concept]
                last_review = reviews[-1]
                interval_idx = min(len(reviews) - 1, len(Config.REPETITION_INTERVALS) - 1)
                due_date = last_review + timedelta(days=Config.REPETITION_INTERVALS[interval_idx])
                priority = 1 if due_date <= datetime.now() else 0
            else:
                priority = 1
            
            prioritized.append((concept, priority))
        
        prioritized.sort(key=lambda x: (-x[1], x[0]))
        return [p[0] for p in prioritized]
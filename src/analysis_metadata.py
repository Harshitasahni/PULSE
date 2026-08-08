"""
analysis_metadata.py
---------------------
Clean metadata structure for analysis results
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class AnalysisMetadata:
    """Metadata classes for window analysis results"""
    forced_reuse: bool = False
    reused_state_id: Optional[int] = None
    new_state_created: bool = False
    similarity_score: float = 0.0
    max_error: float = 0.0
    avg_error: float = 0.0
    
    def is_uncertain(self) -> bool:
        return self.forced_reuse
    
    def should_save_frame(self) -> bool:
        """Check if we should save uncertain frame"""
        return self.forced_reuse  # Save frame when forced to reuse
    
    def to_dict(self) -> dict:
        return {
            'forced_reuse': self.forced_reuse,
            'reused_state_id': self.reused_state_id,
            'new_state_created': self.new_state_created,
            'similarity_score': self.similarity_score,
            'max_error': self.max_error,
            'avg_error': self.avg_error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisMetadata':
        return cls(
            forced_reuse=data.get('forced_reuse', False),
            reused_state_id=data.get('reused_state_id'),
            new_state_created=data.get('new_state_created', False),
            similarity_score=data.get('similarity_score', 0.0),
            max_error=data.get('max_error', 0.0),
            avg_error=data.get('avg_error', 0.0)
        )
    
    @classmethod
    def normal_explanation(cls, state_id: int, max_err: float = 0.0, avg_err: float = 0.0):
        """Window explained normally by existing state"""
        return cls(
            forced_reuse=False,
            new_state_created=False,
            max_error=max_err,
            avg_error=avg_err
        )
    
    @classmethod
    def forced_reuse_explanation(cls, state_id: int, similarity: float, max_err: float = 0.0):
        """Window explained by forced reuse via similarity check"""
        return cls(
            forced_reuse=True,
            reused_state_id=state_id,
            new_state_created=False,
            similarity_score=similarity,
            max_error=max_err
        )
    
    @classmethod
    def new_state_explanation(cls, state_id: int):
        """New state created to explain window"""
        return cls(
            forced_reuse=False,
            new_state_created=True,
            similarity_score=0.0
        )
    
    @classmethod
    def uncertain(cls, max_err: float = 0.0):
        """No state can explain this window"""
        return cls(
            forced_reuse=False,
            new_state_created=False,
            max_error=max_err
        )


class AnalysisResult:
    """Complete result from analyzing a window"""
    
    def __init__(self, state_id: int, metadata: AnalysisMetadata):
        self.state_id = state_id
        self.metadata = metadata
    
    def __iter__(self):
        return iter((self.state_id, self.metadata))
    
    def is_uncertain(self) -> bool:
        return self.state_id == -1 or self.metadata.forced_reuse
    
    def should_save_frame(self) -> bool:
        return self.state_id == -1 or self.metadata.should_save_frame()
    
    def __repr__(self):
        if self.state_id == -1:
            return f"AnalysisResult(state=-1, UNCERTAIN)"
        elif self.state_id == -2:
            return f"AnalysisResult(state=-2, BREAK)"
        elif self.metadata.forced_reuse:
            return f"AnalysisResult(state={self.state_id}, FORCED_REUSE)"
        elif self.metadata.new_state_created:
            return f"AnalysisResult(state={self.state_id}, NEW_STATE)"
        else:
            return f"AnalysisResult(state={self.state_id}, NORMAL)"
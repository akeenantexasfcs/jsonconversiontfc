#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import json
import logging
from typing import Dict, Any, List, Pattern, Optional
import pandas as pd
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum, auto

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Enumeration of possible content types."""
    TABLE = auto()
    CHART = auto()
    TEXT = auto()
    UNKNOWN = auto()

@dataclass
class ProcessingConfig:
    """Configuration parameters for document processing."""
    vertical_gap_threshold: float = 0.05
    confidence_threshold: float = 50.0
    min_group_size: int = 2

@dataclass
class Position:
    """Represents the position of a content block."""
    top: float
    left: float
    width: float = 0.0
    height: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position':
        return cls(
            top=data.get('Top', 0.0),
            left=data.get('Left', 0.0),
            width=data.get('Width', 0.0),
            height=data.get('Height', 0.0)
        )

class DocumentProcessor:
    """Processes Textract JSON output into structured content."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        
        # Precompile regex patterns
        self.table_patterns = [
            re.compile(pattern) for pattern in [
                r'\|\s*\w+',  # Table borders
                r'\+[-+]+\+',  # ASCII table borders
                r'(\d+\s*){3,}',  # Multiple numbers in a row
                r'(\$\s*[\d,]+\s*){2,}',  # Multiple currency values
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b.*\d+'  # Date patterns
            ]
        ]
        
        self.chart_patterns = [
            re.compile(pattern) for pattern in [
                r'(Figure|Fig\.?|Chart)\s*\d+',
                r'(Graph|Plot)\s*\d+',
                r'Diagram\s*\d+'
            ]
        ]

    def identify_content_type(self, block: Dict) -> ContentType:
        """
        Identify the type of content in a block.
        
        Args:
            block: Dictionary containing block information from Textract
            
        Returns:
            ContentType enum indicating the detected content type
        """
        try:
            text = block.get('Text', '')
            if not text:
                return ContentType.UNKNOWN

            # Check table patterns
            if any(pattern.search(text) for pattern in self.table_patterns):
                return ContentType.TABLE

            # Check chart patterns
            if any(pattern.search(text) for pattern in self.chart_patterns):
                return ContentType.CHART

            return ContentType.TEXT

        except Exception as e:
            logger.error(f"Error identifying content type: {e}")
            return ContentType.UNKNOWN

    def group_related_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Group related blocks based on content type and proximity.
        
        Args:
            blocks: List of Textract blocks
            
        Returns:
            List of grouped content blocks
        """
        try:
            grouped_content = []
            current_group = None

            for block in blocks:
                if block.get('BlockType') != 'LINE':
                    continue

                content_type = self.identify_content_type(block)
                position = Position.from_dict(block.get('Geometry', {}).get('BoundingBox', {}))
                confidence = block.get('Confidence', 0)

                # Skip low confidence blocks
                if confidence < self.config.confidence_threshold:
                    logger.warning(f"Skipping low confidence block: {confidence}%")
                    continue

                # Determine if we should start a new group
                start_new_group = (
                    not current_group or
                    content_type != current_group['type'] or
                    abs(position.top - current_group['last_position'].top) > self.config.vertical_gap_threshold
                )

                if start_new_group:
                    if current_group and len(current_group['content']) >= self.config.min_group_size:
                        grouped_content.append(current_group)

                    current_group = {
                        'type': content_type,
                        'content': [],
                        'last_position': position,
                        'page': block.get('Page', 1)
                    }

                if current_group:
                    current_group['content'].append({
                        'text': block.get('Text', ''),
                        'confidence': confidence,
                        'position': position
                    })
                    current_group['last_position'] = position

            # Add the last group if it meets minimum size
            if current_group and len(current_group['content']) >= self.config.min_group_size:
                grouped_content.append(current_group)

            return grouped_content

        except Exception as e:
            logger.error(f"Error grouping blocks: {e}")
            return []

    def process_document(self, data: Dict) -> Dict[str, Any]:
        """
        Process the document into a structured format.
        
        Args:
            data: Raw Textract JSON data
            
        Returns:
            Processed document structure
        """
        try:
            blocks = data.get('Blocks', [])
            grouped_content = self.group_related_blocks(blocks)
            
            # Convert Enum to string for content stats
            content_stats = {
                content_type.name.lower(): sum(1 for g in grouped_content if g['type'] == content_type)
                for content_type in ContentType
            }

            return {
                "metadata": {
                    "processed_date": datetime.now().isoformat(),
                    "total_blocks": len(blocks),
                    "content_distribution": content_stats,
                    "processing_config": {
                        "vertical_gap_threshold": self.config.vertical_gap_threshold,
                        "confidence_threshold": self.config.confidence_threshold,
                        "min_group_size": self.config.min_group_size
                    }
                },
                "content": [
                    {
                        "type": group['type'].name.lower(),
                        "page": group['page'],
                        "text": [item['text'] for item in group['content']],
                        "confidence": sum(item['confidence'] for item in group['content']) / len(group['content']) 
                                    if group['content'] else 0,
                        "position": {
                            "top": min(item['position'].top for item in group['content']),
                            "bottom": max(item['position'].top + item['position'].height 
                                        for item in group['content'])
                        }
                    }
                    for group in grouped_content
                ]
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "metadata": {"error": str(e)},
                "content": []
            }

def main():
    st.title("Enhanced Document Content Processor")
    
    # Sidebar for configuration
    st.sidebar.header("Processing Configuration")
    config = ProcessingConfig(
        vertical_gap_threshold=st.sidebar.slider(
            "Vertical Gap Threshold", 0.01, 0.1, 0.05, 0.01),
        confidence_threshold=st.sidebar.slider(
            "Confidence Threshold", 0.0, 100.0, 50.0, 5.0),
        min_group_size=st.sidebar.number_input(
            "Minimum Group Size", 1, 10, 2)
    )
    
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
    
    if uploaded_file is not None:
        try:
            with st.spinner('Reading file...'):
                content = uploaded_file.read()
                raw_data = json.loads(content)
            
            file_size_mb = len(content)/1024/1024
            st.success(f"File loaded successfully! Size: {file_size_mb:.2f} MB")
            
            processor = DocumentProcessor(config)
            
            with st.spinner('Processing document...'):
                processed_data = processor.process_document(raw_data)
                
                # Content distribution visualization
                st.write("### Content Distribution")
                if processed_data['metadata'].get('content_distribution'):
                    dist_df = pd.DataFrame.from_dict(
                        processed_data['metadata']['content_distribution'], 
                        orient='index', 
                        columns=['count']
                    )
                    st.bar_chart(dist_df)
                
                # Content preview by type
                st.write("### Content Preview")
                for content_type in ContentType:
                    type_name = content_type.name.lower()
                    type_content = [c for c in processed_data['content'] 
                                  if c['type'] == type_name][:2]
                    
                    if type_content:
                        with st.expander(f"{type_name.title()} Content"):
                            for item in type_content:
                                st.text_area(
                                    f"Page {item['page']} (Confidence: {item['confidence']:.2f}%)",
                                    '\n'.join(item['text']),
                                    height=100
                                )
                
                # Direct download button
                st.download_button(
                    label="Download Processed JSON",
                    data=json.dumps(processed_data, indent=2),
                    file_name=f"processed_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {str(e)}")
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()


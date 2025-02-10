#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum, auto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    TABLE = auto()
    CHART = auto()
    TEXT = auto()
    HEADER = auto()

@dataclass
class ProcessingConfig:
    vertical_gap_threshold: float = 0.02
    horizontal_gap_threshold: float = 0.01
    confidence_threshold: float = 50.0
    min_group_size: int = 2
    table_row_threshold: float = 0.015

@dataclass
class Position:
    top: float
    left: float
    width: float = 0.0
    height: float = 0.0
    
    @property
    def right(self) -> float:
        return self.left + self.width
    
    @property
    def bottom(self) -> float:
        return self.top + self.height

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position':
        return cls(
            top=data.get('Top', 0.0),
            left=data.get('Left', 0.0),
            width=data.get('Width', 0.0),
            height=data.get('Height', 0.0)
        )

class DocumentProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.table_patterns = [
            re.compile(pattern) for pattern in [
                r'(\d+[\s,.]){3,}',  # Multiple numbers
                r'(\$[\s\d,.]+\s*){2,}',  # Currency values
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b.*?\d+',  # Dates
                r'^\s*[\d.]+\s*$'  # Standalone numbers
            ]
        ]

    def identify_content_type(self, block: Dict, blocks: List[Dict]) -> ContentType:
        text = block.get('Text', '').strip()
        if not text:
            return ContentType.TEXT
            
        position = Position.from_dict(block.get('Geometry', {}).get('BoundingBox', {}))
        
        # Check if part of table
        if any(pattern.search(text) for pattern in self.table_patterns):
            nearby_blocks = [b for b in blocks if abs(Position.from_dict(
                b.get('Geometry', {}).get('BoundingBox', {})
            ).top - position.top) < self.config.table_row_threshold]
            if len(nearby_blocks) > 2:  # At least 3 aligned items suggest a table row
                return ContentType.TABLE
        
        # Check if header
        if len(text) < 50 and position.top < 0.3:
            return ContentType.HEADER
            
        return ContentType.TEXT

    def process_table(self, blocks: List[Dict]) -> Dict[str, Any]:
        rows = []
        current_row = []
        last_top = None
        
        # Sort blocks by vertical position, then horizontal
        sorted_blocks = sorted(blocks, key=lambda x: (
            x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0),
            x.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)
        ))
        
        for block in sorted_blocks:
            pos = Position.from_dict(block.get('Geometry', {}).get('BoundingBox', {}))
            
            # New row detection
            if last_top is not None and abs(pos.top - last_top) > self.config.table_row_threshold:
                if current_row:
                    rows.append(current_row)
                    current_row = []
            
            current_row.append({
                'text': block.get('Text', ''),
                'position': {'left': pos.left, 'top': pos.top},
                'confidence': block.get('Confidence', 0)
            })
            last_top = pos.top
        
        if current_row:
            rows.append(current_row)
        
        # Determine column structure
        if rows:
            columns = len(rows[0])
            return {
                'rows': rows,
                'structure': {'columns': columns, 'row_count': len(rows)},
                'position': {
                    'top': min(block.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0) for block in blocks),
                    'bottom': max(block.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0) + 
                                block.get('Geometry', {}).get('BoundingBox', {}).get('Height', 0) for block in blocks)
                }
            }
        return None

    def process_document(self, data: Dict) -> Dict[str, Any]:
        try:
            blocks = data.get('Blocks', [])
            tables = []
            text_blocks = []
            headers = []
            current_table_blocks = []
            
            for block in blocks:
                if block.get('BlockType') != 'LINE':
                    continue
                    
                content_type = self.identify_content_type(block, blocks)
                pos = Position.from_dict(block.get('Geometry', {}).get('BoundingBox', {}))
                
                if content_type == ContentType.TABLE:
                    current_table_blocks.append(block)
                elif current_table_blocks:
                    # Process completed table
                    table = self.process_table(current_table_blocks)
                    if table:
                        tables.append(table)
                    current_table_blocks = []
                    
                if content_type == ContentType.HEADER:
                    headers.append({
                        'text': block.get('Text', ''),
                        'position': {'top': pos.top, 'left': pos.left},
                        'confidence': block.get('Confidence', 0)
                    })
                elif content_type == ContentType.TEXT:
                    text_blocks.append({
                        'text': block.get('Text', ''),
                        'position': {'top': pos.top, 'left': pos.left},
                        'confidence': block.get('Confidence', 0),
                        'page': block.get('Page', 1)
                    })
            
            # Process final table if exists
            if current_table_blocks:
                table = self.process_table(current_table_blocks)
                if table:
                    tables.append(table)

            return {
                "metadata": {
                    "processed_date": datetime.now().isoformat(),
                    "total_blocks": len(blocks),
                    "tables_found": len(tables)
                },
                "content": {
                    "tables": tables,
                    "headers": headers,
                    "text_blocks": text_blocks
                }
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"metadata": {"error": str(e)}, "content": {}}

def main():
    st.title("Document Content Processor")
    
    config = ProcessingConfig(
        vertical_gap_threshold=st.sidebar.slider("Vertical Gap", 0.01, 0.05, 0.02, 0.005),
        confidence_threshold=st.sidebar.slider("Confidence Threshold", 0.0, 100.0, 50.0, 5.0)
    )
    
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
    
    if uploaded_file:
        try:
            raw_data = json.loads(uploaded_file.read())
            processor = DocumentProcessor(config)
            
            with st.spinner('Processing document...'):
                processed_data = processor.process_document(raw_data)
                
                if processed_data['content'].get('tables'):
                    st.write(f"Found {len(processed_data['content']['tables'])} tables")
                    for i, table in enumerate(processed_data['content']['tables']):
                        with st.expander(f"Table {i+1}"):
                            if table['rows']:
                                df = pd.DataFrame([
                                    [cell['text'] for cell in row]
                                    for row in table['rows']
                                ])
                                st.dataframe(df)
                
                st.download_button(
                    "Download Processed JSON",
                    data=json.dumps(processed_data, indent=2),
                    file_name=f"processed_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()


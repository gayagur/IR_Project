from flask import Flask, request, render_template_string, send_from_directory
import requests
import time
from pathlib import Path

app = Flask(__name__)

@app.route('/static/assets/<path:filename>')
def serve_assets(filename):
    assets_dir = Path(__file__).parent / 'assets'
    return send_from_directory(str(assets_dir), filename)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WikiSearch | Wikipedia Search Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root {
            /* BACKGROUND LAYERS - Depth hierarchy */
            --bg-base: #0a0a0c;           /* Deepest - page background */
            --bg-layer-1: #111113;        /* Section backgrounds */
            --bg-layer-2: #18181b;       /* Card backgrounds */
            --bg-layer-3: #1f1f23;       /* Elevated cards / hover */
            
            /* SURFACE CARDS */
            --surface-card: #18181b;
            --surface-elevated: #1f1f23;
            --surface-hover: #252529;
            
            /* BORDERS - Subtle, low opacity */
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-soft: rgba(255, 255, 255, 0.08);
            --border-medium: rgba(255, 255, 255, 0.12);
            --border-strong: rgba(255, 255, 255, 0.16);
            
            /* HARMONIZED ACCENT PALETTE - Premium, controlled */
            /* Teal - Primary action and focus */
            --accent-teal: #2dd4bf;
            --teal-soft: rgba(45, 212, 191, 0.10);
            --teal-chip: rgba(45, 212, 191, 0.14);
            --teal-glow: rgba(45, 212, 191, 0.08);
            
            /* Indigo - Analytics and semantic */
            --accent-indigo: #818cf8;
            --indigo-soft: rgba(129, 140, 248, 0.10);
            --indigo-chip: rgba(129, 140, 248, 0.14);
            --indigo-glow: rgba(129, 140, 248, 0.08);
            
            /* Amber - Popularity and engagement */
            --accent-amber: #fbbf24;
            --amber-soft: rgba(251, 191, 36, 0.10);
            --amber-chip: rgba(251, 191, 36, 0.14);
            --amber-glow: rgba(251, 191, 36, 0.08);
            
            /* LEGACY - Primary/secondary for backward compatibility */
            --accent-primary: var(--accent-teal);
            --accent-primary-hover: #20b2a3;
            --accent-primary-light: var(--teal-soft);
            --accent-primary-glow: var(--teal-glow);
            --accent-secondary: var(--accent-indigo);
            --accent-secondary-light: var(--indigo-soft);
            
            /* TEXT HIERARCHY */
            --text-primary: #fafafa;       /* High contrast - headings */
            --text-secondary: #a1a1aa;    /* Body text */
            --text-tertiary: #71717a;     /* Metadata, hints */
            --text-disabled: #52525b;     /* Disabled states */
            
            /* SHADOWS - Soft depth */
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.6);
            
            /* LEGACY - Backward compatibility */
            --bg: var(--bg-base);
            --bg-secondary: var(--bg-layer-2);
            --bg-tertiary: var(--bg-layer-3);
            --border: var(--border-soft);
            --border-light: var(--border-medium);
            --text: var(--text-primary);
            --text-secondary: var(--text-secondary);
            --text-muted: var(--text-tertiary);
            --accent: var(--accent-primary);
            --accent-secondary: var(--accent-primary-hover);
            --accent-glow: var(--accent-primary-glow);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-base);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        /* Subtle grid background - very subtle */
        .bg-grid {
            position: fixed;
            inset: 0;
            z-index: -1;
            background-image: 
                linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
            background-size: 64px 64px;
            opacity: 0.4;
        }
        
        /* Subtle gradient overlay - minimal */
        .bg-gradient {
            position: fixed;
            inset: 0;
            z-index: -1;
            overflow: hidden;
            opacity: 0.3;
        }
        
        .bg-gradient::before {
            content: '';
            position: absolute;
            top: -20%;
            left: 50%;
            transform: translateX(-50%);
            width: 800px;
            height: 600px;
            background: radial-gradient(ellipse, var(--teal-glow) 0%, transparent 70%);
            filter: blur(80px);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0;
        }
        
        /* Section spacing with layered backgrounds */
        .section {
            padding: 80px 24px;
            background: var(--bg-layer-1);
            position: relative;
        }
        
        .section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        }
        
        .section-sm {
            padding: 48px 24px;
            background: var(--bg-base);
        }
        
        /* Alternate section backgrounds for depth */
        .section:nth-child(even) {
            background: var(--bg-base);
        }
        
        .section:nth-child(odd) {
            background: var(--bg-layer-1);
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 100px 24px 80px;
            position: relative;
        }
        
        .hero-content {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .hero-main {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 60px;
            align-items: center;
            margin-bottom: 60px;
        }
        
        .hero-text {
            text-align: left;
        }
        
        .hero-title {
            font-size: 56px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 20px;
            color: var(--text-primary);
            letter-spacing: -1.5px;
        }
        
        .hero-subtitle {
            font-size: 22px;
            color: var(--text-secondary);
            margin-bottom: 32px;
            line-height: 1.5;
        }
        
        .hero-features {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .hero-feature {
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--text-secondary);
            font-size: 15px;
        }
        
        .hero-feature i {
            color: var(--accent-teal);
            font-size: 18px;
            width: 24px;
        }
        
        .hero-video {
            position: relative;
        }
        
        .video-wrapper {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            background: var(--surface-card);
            box-shadow: var(--shadow-lg);
            position: relative;
        }
        
        .video-wrapper::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, var(--teal-glow), transparent);
            z-index: 1;
            pointer-events: none;
            opacity: 0.3;
        }
        
        .video-wrapper video {
            width: 100%;
            display: block;
            position: relative;
            z-index: 0;
        }
        
        .video-caption {
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 10, 12, 0.85);
            backdrop-filter: blur(10px);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            z-index: 2;
            border: 1px solid var(--border-subtle);
        }
        
        /* Product Highlights */
        .highlights {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 60px;
        }
        
        .highlight-card {
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 28px 24px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .highlight-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-teal);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        /* Teal variant */
        .highlight-card.teal::before {
            background: var(--accent-teal);
        }
        
        .highlight-card.teal .highlight-icon {
            background: var(--teal-chip);
            color: var(--accent-teal);
        }
        
        /* Indigo variant */
        .highlight-card.indigo::before {
            background: var(--accent-indigo);
        }
        
        .highlight-card.indigo .highlight-icon {
            background: var(--indigo-chip);
            color: var(--accent-indigo);
        }
        
        /* Amber variant */
        .highlight-card.amber::before {
            background: var(--accent-amber);
        }
        
        .highlight-card.amber .highlight-icon {
            background: var(--amber-chip);
            color: var(--accent-amber);
        }
        
        .highlight-card:hover {
            background: var(--surface-elevated);
            border-color: var(--border-soft);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .highlight-card:hover::before {
            transform: scaleX(1);
        }
        
        .highlight-icon {
            width: 56px;
            height: 56px;
            margin: 0 auto 16px;
            background: var(--teal-chip);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: var(--accent-teal);
        }
        
        .highlight-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
        }
        
        .highlight-label {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.4;
        }
        
        /* Section Titles */
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .section-title i {
            color: var(--accent-teal);
            font-size: 16px;
        }
        
        /* Search Dashboard */
        .search-dashboard {
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: var(--shadow-sm);
        }
        
        .search-card {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
        }
        
        .search-input-wrapper {
            position: relative;
            margin-bottom: 24px;
        }
        
        .search-input {
            width: 100%;
            padding: 16px 20px 16px 48px;
            font-size: 15px;
            font-family: inherit;
            background: var(--bg-layer-3);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            color: var(--text-primary);
            outline: none;
            transition: all 0.2s;
        }
        
        .search-input:focus {
            background: var(--surface-card);
            border-color: var(--accent-teal);
            box-shadow: 0 0 0 3px var(--teal-glow);
        }
        
        .search-input::placeholder {
            color: var(--text-tertiary);
        }
        
        .search-input-icon {
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-tertiary);
            font-size: 14px;
        }
        
        /* Methods - Segmented Control Style with Color Logic */
        .methods-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }
        
        .methods-wrapper {
            background: var(--bg-layer-3);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 6px;
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 6px;
            margin-bottom: 24px;
        }
        
        .method-btn {
            position: relative;
        }
        
        .method-btn input {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }
        
        .method-btn label {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            padding: 12px 8px;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }
        
        .method-btn label:hover {
            background: var(--surface-hover);
        }
        
        /* Keyword-based methods (Main, Body, Title, Anchor) - Teal */
        .method-btn[data-type="keyword"] input:checked + label {
            background: var(--accent-teal);
            color: white;
        }
        
        /* Semantic method - Indigo */
        .method-btn[data-type="semantic"] input:checked + label {
            background: var(--accent-indigo);
            color: white;
        }
        
        /* Popularity-based (PageRank, Views) - Amber */
        .method-btn[data-type="popularity"] input:checked + label {
            background: var(--accent-amber);
            color: white;
        }
        
        /* Default checked state - Teal (fallback) */
        .method-btn input:checked + label {
            background: var(--accent-teal);
            color: white;
        }
        
        .method-btn i {
            font-size: 16px;
            color: var(--text-tertiary);
            transition: color 0.2s;
        }
        
        .method-btn input:checked + label i,
        .method-btn label:hover i {
            color: inherit;
        }
        
        .method-btn span {
            font-size: 11px;
            color: var(--text-secondary);
            font-weight: 500;
            transition: color 0.2s;
        }
        
        .method-btn input:checked + label span {
            color: white;
        }
        
        .method-description {
            font-size: 11px;
            color: var(--text-tertiary);
            margin-top: 4px;
            text-align: center;
        }
        
        /* Search button - Primary action (Teal) */
        .search-btn {
            width: 100%;
            padding: 14px 24px;
            font-size: 14px;
            font-weight: 600;
            font-family: inherit;
            background: var(--accent-teal);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s;
            box-shadow: var(--shadow-sm);
        }
        
        .search-btn:hover {
            background: #20b2a3;
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .search-btn:active {
            transform: translateY(0);
        }
        
        /* Analytics Cards */
        .analytics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .analytics-card {
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-sm);
        }
        
        .analytics-card:hover {
            background: var(--surface-elevated);
            border-color: var(--border-soft);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .analytics-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-secondary-light);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: var(--accent-secondary);
            flex-shrink: 0;
        }
        
        .analytics-content {
            flex: 1;
        }
        
        .analytics-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
            line-height: 1;
        }
        
        .analytics-label {
            font-size: 12px;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Stats Cards - Updated */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat {
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat:hover {
            background: var(--surface-elevated);
            border-color: var(--border-soft);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .stat-icon {
            width: 40px;
            height: 40px;
            background: var(--indigo-chip);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: var(--accent-indigo);
        }
        
        /* Stat card color variants */
        .stat.teal .stat-icon {
            background: var(--teal-chip);
            color: var(--accent-teal);
        }
        
        .stat.indigo .stat-icon {
            background: var(--indigo-chip);
            color: var(--accent-indigo);
        }
        
        .stat.amber .stat-icon {
            background: var(--amber-chip);
            color: var(--accent-amber);
        }
        
        /* Optional accent bar at top */
        .stat::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: transparent;
            border-radius: 16px 16px 0 0;
        }
        
        .stat.teal::before {
            background: var(--accent-teal);
        }
        
        .stat.indigo::before {
            background: var(--accent-indigo);
        }
        
        .stat.amber::before {
            background: var(--accent-amber);
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Results */
        .results-card {
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .results-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-layer-3);
        }
        
        .results-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .results-title span {
            color: var(--accent-teal);
        }
        
        .results-method {
            font-size: 12px;
            color: var(--text-secondary);
            padding: 4px 10px;
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
        }
        
        .results-list {
            padding: 8px;
        }
        
        .result-item {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 20px;
            border-radius: 12px;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
            border: 1px solid transparent;
            margin-bottom: 8px;
        }
        
        .result-item:hover {
            background: var(--surface-hover);
            border-color: var(--border-soft);
            transform: translateX(2px);
        }
        
        /* Rank #1 - Teal */
        .result-item:nth-child(1) {
            background: var(--teal-soft);
            border-color: var(--border-soft);
            position: relative;
        }
        
        .result-item:nth-child(1)::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--accent-teal);
            border-radius: 12px 0 0 12px;
        }
        
        .result-item:nth-child(1) .result-rank {
            background: var(--accent-teal);
            border: none;
            color: white;
            box-shadow: var(--shadow-md);
        }
        
        /* Rank #2 - Indigo */
        .result-item:nth-child(2) {
            background: var(--indigo-soft);
            border-color: var(--border-soft);
            position: relative;
        }
        
        .result-item:nth-child(2)::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--accent-indigo);
            border-radius: 12px 0 0 12px;
        }
        
        .result-item:nth-child(2) .result-rank {
            background: var(--accent-indigo);
            border: none;
            color: white;
            box-shadow: var(--shadow-sm);
        }
        
        /* Rank #3 - Amber */
        .result-item:nth-child(3) {
            background: var(--amber-soft);
            border-color: var(--border-soft);
            position: relative;
        }
        
        .result-item:nth-child(3)::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--accent-amber);
            border-radius: 12px 0 0 12px;
        }
        
        .result-item:nth-child(3) .result-rank {
            background: var(--accent-amber);
            border: none;
            color: white;
            box-shadow: var(--shadow-sm);
        }
        
        .result-rank {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-layer-3);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            color: var(--text-secondary);
            flex-shrink: 0;
            transition: all 0.3s ease;
        }
        
        .result-content {
            flex: 1;
            min-width: 0;
        }
        
        .result-title {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .result-meta {
            font-size: 12px;
            color: var(--text-tertiary);
        }
        
        .result-arrow {
            color: var(--text-tertiary);
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .result-item:hover .result-arrow {
            color: var(--accent-teal);
            transform: translateX(4px);
        }
        
        /* Pagination */
        .pagination {
            display: flex;
            justify-content: center;
            gap: 6px;
            padding: 16px;
            border-top: 1px solid var(--border-subtle);
        }
        
        .pagination a, .pagination span {
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 500;
            background: var(--bg-layer-3);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            color: var(--text-secondary);
            text-decoration: none;
            transition: all 0.2s;
        }
        
        .pagination a:hover {
            background: var(--surface-hover);
            border-color: var(--border-soft);
            color: var(--text-primary);
        }
        
        .pagination .current {
            background: var(--accent-teal);
            border-color: var(--accent-teal);
            color: white;
        }
        
        .pagination .disabled {
            opacity: 0.3;
            pointer-events: none;
            color: var(--text-disabled);
        }
        
        /* No results */
        .no-results {
            padding: 64px 24px;
            text-align: center;
        }
        
        .no-results i {
            font-size: 48px;
            color: var(--text-tertiary);
            margin-bottom: 16px;
        }
        
        .no-results h3 {
            font-size: 16px;
            color: var(--text-primary);
            margin-bottom: 8px;
        }
        
        .no-results p {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 48px 24px;
            color: var(--text-tertiary);
            font-size: 13px;
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-bottom: 16px;
        }
        
        .footer-links a {
            color: var(--text-secondary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: color 0.2s;
        }
        
        .footer-links a:hover {
            color: var(--accent-teal);
        }
        
        .footer-credits span {
            color: var(--accent-teal);
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .section {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .highlight-card {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .highlight-card:nth-child(1) { animation-delay: 0.1s; }
        .highlight-card:nth-child(2) { animation-delay: 0.2s; }
        .highlight-card:nth-child(3) { animation-delay: 0.3s; }
        .highlight-card:nth-child(4) { animation-delay: 0.4s; }
        
        /* Section Dividers */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            margin: 40px 0;
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .hero-main {
                grid-template-columns: 1fr;
                gap: 40px;
            }
            
            .hero-text {
                text-align: center;
            }
            
            .highlights {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .section {
                padding: 60px 20px;
            }
            
            .section-sm {
                padding: 40px 20px;
            }
            
            .hero {
                padding: 60px 20px 40px;
            }
            
            .hero-title {
                font-size: 42px;
            }
            
            .hero-subtitle {
                font-size: 18px;
            }
            
            .methods-wrapper {
                grid-template-columns: repeat(3, 1fr);
            }
            
            .highlights {
                grid-template-columns: 1fr;
            }
            
            .search-dashboard {
                padding: 28px;
            }
            
            .stats {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 480px) {
            .hero-title {
                font-size: 32px;
            }
            
            .methods-wrapper {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .search-dashboard {
                padding: 20px;
            }
        }
    </style>
    <script>
        // Smooth scroll behavior
        document.documentElement.style.scrollBehavior = 'smooth';
        
        // Animate counters on stats
        function animateCounter(element, target, duration = 1000) {
            const start = 0;
            const increment = target / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    element.textContent = target;
                    clearInterval(timer);
                } else {
                    element.textContent = Math.floor(current);
                }
            }, 16);
        }
        
        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);
        
        // Observe sections on load
        document.addEventListener('DOMContentLoaded', () => {
            const sections = document.querySelectorAll('.section, .section-sm');
            sections.forEach(section => {
                section.style.opacity = '0';
                section.style.transform = 'translateY(20px)';
                section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                observer.observe(section);
            });
            
            // Animate stat counters if they exist
            const statValues = document.querySelectorAll('.stat-value');
            statValues.forEach(stat => {
                const text = stat.textContent.trim();
                const number = parseFloat(text);
                if (!isNaN(number) && number > 0 && number < 10000) {
                    stat.textContent = '0';
                    setTimeout(() => animateCounter(stat, number), 200);
                }
            });
        });
    </script>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="bg-gradient"></div>
    
    <div class="container">
        <!-- HERO SECTION -->
        <section class="hero section">
            <div class="hero-content">
                <div class="hero-main">
                    <div class="hero-text">
                        <h1 class="hero-title">WikiSearch</h1>
                        <p class="hero-subtitle">A High-Performance Search Engine over 6.3M Wikipedia Articles</p>
                        <div class="hero-features">
                            <div class="hero-feature">
                                <i class="fas fa-layer-group"></i>
                                <span>Multi-signal ranking (BM25, TF-IDF, PageRank, GloVe)</span>
                            </div>
                            <div class="hero-feature">
                                <i class="fas fa-brain"></i>
                                <span>Semantic understanding beyond keywords</span>
                            </div>
                            <div class="hero-feature">
                                <i class="fas fa-database"></i>
                                <span>Built on real Wikipedia scale data</span>
                            </div>
                        </div>
                    </div>
                    <div class="hero-video">
                        <div class="video-wrapper">
                            <video autoplay loop muted playsinline>
                                <source src="/static/assets/ui.mp4" type="video/mp4">
                            </video>
                            <div class="video-caption">Live Search Demo</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- PRODUCT HIGHLIGHTS -->
        <section class="section-sm">
            <div class="highlights">
                <div class="highlight-card teal">
                    <div class="highlight-icon">
                        <i class="fas fa-database"></i>
                    </div>
                    <div class="highlight-value">6.3M</div>
                    <div class="highlight-label">Articles Indexed</div>
                </div>
                <div class="highlight-card indigo">
                    <div class="highlight-icon">
                        <i class="fas fa-layer-group"></i>
                    </div>
                    <div class="highlight-value">6</div>
                    <div class="highlight-label">Ranking Signals</div>
                </div>
                <div class="highlight-card amber">
                    <div class="highlight-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="highlight-value">GloVe</div>
                    <div class="highlight-label">Semantic Search</div>
                </div>
                <div class="highlight-card teal">
                    <div class="highlight-icon">
                        <i class="fas fa-bolt"></i>
                    </div>
                    <div class="highlight-value">&lt;1s</div>
                    <div class="highlight-label">Query Time</div>
                </div>
            </div>
        </section>
        
        <!-- SEARCH DASHBOARD -->
        <section class="section-sm">
            <div class="search-dashboard">
                <div class="section-title">
                    <i class="fas fa-sliders-h"></i>
                    <span>Search Configuration</span>
                </div>
                <div class="search-card">
            <form method="GET" action="/">
                <div class="search-input-wrapper">
                    <i class="fas fa-search search-input-icon"></i>
                    <input 
                        type="text" 
                        name="query" 
                        class="search-input" 
                        placeholder="Search articles..." 
                        value="{{ query or '' }}"
                        autofocus
                    >
                </div>
                
                <div class="methods-label">Search Method</div>
                <div class="methods-wrapper">
                    <div class="method-btn" data-type="keyword">
                        <input type="radio" name="method" id="main" value="main" {{ 'checked' if method == 'main' or not method else '' }}>
                        <label for="main">
                            <i class="fas fa-bolt"></i>
                            <span>Main</span>
                        </label>
                    </div>
                    <div class="method-btn" data-type="keyword">
                        <input type="radio" name="method" id="body" value="body" {{ 'checked' if method == 'body' else '' }}>
                        <label for="body">
                            <i class="fas fa-file-alt"></i>
                            <span>TF-IDF</span>
                        </label>
                    </div>
                    <div class="method-btn" data-type="keyword">
                        <input type="radio" name="method" id="title" value="title" {{ 'checked' if method == 'title' else '' }}>
                        <label for="title">
                            <i class="fas fa-heading"></i>
                            <span>Title</span>
                        </label>
                    </div>
                    <div class="method-btn" data-type="keyword">
                        <input type="radio" name="method" id="anchor" value="anchor" {{ 'checked' if method == 'anchor' else '' }}>
                        <label for="anchor">
                            <i class="fas fa-link"></i>
                            <span>Anchor</span>
                        </label>
                    </div>
                    <div class="method-btn" data-type="popularity">
                        <input type="radio" name="method" id="pagerank" value="pagerank" {{ 'checked' if method == 'pagerank' else '' }}>
                        <label for="pagerank">
                            <i class="fas fa-chart-line"></i>
                            <span>PageRank</span>
                        </label>
                    </div>
                    <div class="method-btn" data-type="popularity">
                        <input type="radio" name="method" id="pageviews" value="pageviews" {{ 'checked' if method == 'pageviews' else '' }}>
                        <label for="pageviews">
                            <i class="fas fa-eye"></i>
                            <span>Views</span>
                        </label>
                    </div>
                </div>
                
                <button type="submit" class="search-btn">
                    <i class="fas fa-search"></i>
                    Search
                </button>
            </form>
            </div>
        </div>
        </section>
        
        {% if query %}
        <!-- RESULTS ANALYTICS -->
        <section class="section-sm">
            <div class="section-title">
                <i class="fas fa-chart-line"></i>
                <span>Query Analytics</span>
            </div>
            <div class="stats">
                <div class="stat teal">
                    <div class="stat-icon">
                        <i class="fas fa-list-ul"></i>
                    </div>
                    <div class="stat-value">{{ total_results }}</div>
                    <div class="stat-label">Total Results Retrieved</div>
                </div>
                <div class="stat indigo">
                    <div class="stat-icon">
                        <i class="fas fa-stopwatch"></i>
                    </div>
                    <div class="stat-value">{{ "%.2f"|format(time) }}s</div>
                    <div class="stat-label">Query Latency</div>
                </div>
                <div class="stat amber">
                    <div class="stat-icon">
                        <i class="fas fa-cog"></i>
                    </div>
                    <div class="stat-value">{{ method_name }}</div>
                    <div class="stat-label">Ranking Method Used</div>
                </div>
            </div>
        </section>
        
        <!-- RESULTS LIST -->
        <section class="section-sm">
        
        <div class="results-card">
            <div class="results-header">
                <div class="results-title">Results for "<span>{{ query }}</span>"</div>
                <div class="results-method">{{ method_name }}</div>
            </div>
            
            {% if results %}
            <div class="results-list">
                {% for doc_id, title in results %}
                <a href="https://en.wikipedia.org/?curid={{ doc_id }}" target="_blank" class="result-item">
                    <div class="result-rank">{{ (page - 1) * 20 + loop.index }}</div>
                    <div class="result-content">
                        <div class="result-title">{{ title or 'Untitled' }}</div>
                        <div class="result-meta">ID: {{ doc_id }}</div>
                    </div>
                    <i class="fas fa-arrow-right result-arrow"></i>
                </a>
                {% endfor %}
            </div>
            
            {% if total_pages > 1 %}
            <div class="pagination">
                {% if page > 1 %}
                    <a href="?query={{ query }}&method={{ method }}&page={{ page - 1 }}">Prev</a>
                {% else %}
                    <span class="disabled">Prev</span>
                {% endif %}
                
                {% for p in range(1, total_pages + 1) %}
                    {% if p == page %}
                        <span class="current">{{ p }}</span>
                    {% elif p <= 2 or p > total_pages - 1 or (p >= page - 1 and p <= page + 1) %}
                        <a href="?query={{ query }}&method={{ method }}&page={{ p }}">{{ p }}</a>
                    {% elif p == 3 and page > 4 %}
                        <span>...</span>
                    {% endif %}
                {% endfor %}
                
                {% if page < total_pages %}
                    <a href="?query={{ query }}&method={{ method }}&page={{ page + 1 }}">Next</a>
                {% else %}
                    <span class="disabled">Next</span>
                {% endif %}
            </div>
            {% endif %}
            
            {% else %}
            <div class="no-results">
                <i class="fas fa-search"></i>
                <h3>No results found</h3>
                <p>Try different keywords</p>
            </div>
            {% endif %}
        </div>
        </section>
        {% endif %}
        
        <!-- FOOTER -->
        <section class="section">
            <footer class="footer">
            <div class="footer-links">
                <a href="https://github.com/gayagur/IR_Project" target="_blank">
                    <i class="fab fa-github"></i> GitHub
                </a>
                <a href="https://en.wikipedia.org" target="_blank">
                    <i class="fab fa-wikipedia-w"></i> Wikipedia
                </a>
            </div>
            <div class="footer-credits">
                Built by <span>Gaya Gur & Matias Guernik</span> · IR Course 2025/2026
            </div>
        </footer>
        </section>
    </div>
</body>
</html>
'''

METHOD_NAMES = {
    'main': 'Main Search',
    'body': 'Body TF-IDF',
    'title': 'Title',
    'anchor': 'Anchor',
    'pagerank': 'PageRank',
    'pageviews': 'Page Views'
}

ENDPOINTS = {
    'main': '/search',
    'body': '/search_body',
    'title': '/search_title',
    'anchor': '/search_anchor',
}

@app.route('/')
def search():
    query = request.args.get('query', '').strip()
    method = request.args.get('method', 'main')
    page = int(request.args.get('page', 1))
    results = []
    all_results = []
    search_time = 0
    total_results = 0
    
    if query:
        try:
            start = time.time()
            
            if method == 'pagerank':
                response = requests.get('http://localhost:8080/search', params={'query': query}, timeout=60)
                if response.status_code == 200:
                    main_results = response.json()
                    doc_ids = [doc_id for doc_id, title in main_results]
                    pr_response = requests.post('http://localhost:8080/get_pagerank', json=doc_ids, timeout=30)
                    if pr_response.status_code == 200:
                        pr_scores = pr_response.json()
                        combined = list(zip(main_results, pr_scores))
                        combined.sort(key=lambda x: x[1], reverse=True)
                        all_results = [item[0] for item in combined][:100]
                        
            elif method == 'pageviews':
                response = requests.get('http://localhost:8080/search', params={'query': query}, timeout=60)
                if response.status_code == 200:
                    main_results = response.json()
                    doc_ids = [doc_id for doc_id, title in main_results]
                    pv_response = requests.post('http://localhost:8080/get_pageview', json=doc_ids, timeout=30)
                    if pv_response.status_code == 200:
                        pv_scores = pv_response.json()
                        combined = list(zip(main_results, pv_scores))
                        combined.sort(key=lambda x: x[1], reverse=True)
                        all_results = [item[0] for item in combined][:100]
            else:
                endpoint = ENDPOINTS.get(method, '/search')
                response = requests.get(f'http://localhost:8080{endpoint}', params={'query': query}, timeout=60)
                if response.status_code == 200:
                    all_results = response.json()[:100]
                    
            search_time = time.time() - start
            total_results = len(all_results)
            
            results_per_page = 20
            start_idx = (page - 1) * results_per_page
            end_idx = start_idx + results_per_page
            results = all_results[start_idx:end_idx]
            
        except Exception as e:
            print(f"Error: {e}")
    
    results_per_page = 20
    total_pages = (total_results + results_per_page - 1) // results_per_page if total_results > 0 else 0
    
    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        method=method,
        method_name=METHOD_NAMES.get(method, 'Main Search'),
        results=results,
        total_results=total_results,
        time=search_time,
        page=page,
        total_pages=total_pages
    )

if __name__ == '__main__':
    print("=" * 50)
    print("  WikiSearch - Starting on port 8082")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8082, debug=False)
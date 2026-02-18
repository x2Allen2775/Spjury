#!/usr/bin/env python3
"""
Cricket Batting Biomechanics Analysis Module
Adapted from batting_analyzer.py (UnifiedCricketAnalyzer)
No GUI picker - video path passed as parameter
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
from collections import deque
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


class CricketAnalysisConfig:
    def __init__(self):
        self.min_detection_confidence = 0.6
        self.min_tracking_confidence = 0.6
        self.batting_injury_thresholds = {
            'knee_angle_min': 100.0,
            'knee_angle_low': 120.0,
            'hip_angle_min': 110.0,
            'hip_angle_low': 130.0,
            'weight_imbalance_high': 80.0,
        }


class ActivityDetector:
    @staticmethod
    def detect_activity(landmarks_history, frame_count=60):
        if len(landmarks_history) < 30:
            return 'UNKNOWN'
        wrist_movements = []
        hip_movements = []
        for i in range(1, min(len(landmarks_history), frame_count)):
            prev = landmarks_history[i-1]
            curr = landmarks_history[i]
            if prev and curr:
                wrist_move = abs(curr['r_wrist'].y - prev['r_wrist'].y)
                hip_move = abs(curr['hip_x'] - prev['hip_x'])
                wrist_movements.append(wrist_move)
                hip_movements.append(hip_move)
        if not wrist_movements:
            return 'UNKNOWN'
        avg_hip = statistics.mean(hip_movements)
        max_wrist = max(wrist_movements)
        if max_wrist > 0.15 and avg_hip > 0.01:
            return 'BOWLING'
        elif max_wrist < 0.10 and avg_hip < 0.008:
            return 'BATTING'
        else:
            return 'BOWLING' if avg_hip > 0.008 else 'BATTING'


class BattingAnalyzer:
    def __init__(self, config):
        self.config = config
        self.front_weight_history = []
        self.back_weight_history = []
        self.risk_scores = []
        self.knee_angles = []
        self.hip_angles = []
        self.frames_analyzed = 0

    def analyze_frame(self, landmarks, frame_w, frame_h):
        results = {}
        lh = landmarks['l_hip']; rh = landmarks['r_hip']
        lk = landmarks['l_knee']; rk = landmarks['r_knee']
        la = landmarks['l_ankle']; ra = landmarks['r_ankle']
        ls = landmarks['l_shoulder']; rs = landmarks['r_shoulder']
        lf = landmarks.get('l_foot', la); rf = landmarks.get('r_foot', ra)
        com_x = (lh.x + rh.x) / 2
        dist_left = abs(com_x - lf.x); dist_right = abs(com_x - rf.x)
        total = dist_left + dist_right + 1e-6
        left_w = (1 - dist_left / total) * 100; right_w = (1 - dist_right / total) * 100
        scale = 100 / (left_w + right_w + 1e-6)
        left_w *= scale; right_w *= scale
        front_weight = right_w; back_weight = left_w
        results['front_weight'] = front_weight; results['back_weight'] = back_weight
        self.front_weight_history.append(front_weight); self.back_weight_history.append(back_weight)
        left_knee = self._calc_angle(lh, lk, la); right_knee = self._calc_angle(rh, rk, ra)
        left_hip = self._calc_angle(ls, lh, lk); right_hip = self._calc_angle(rs, rh, rk)
        avg_knee = (left_knee + right_knee) / 2; avg_hip = (left_hip + right_hip) / 2
        results['knee_angle'] = avg_knee; results['hip_angle'] = avg_hip
        self.knee_angles.append(avg_knee); self.hip_angles.append(avg_hip)
        risk_score, risk_factors = self._assess_injury_risk(avg_knee, avg_hip, front_weight, back_weight)
        results['risk_score'] = risk_score; results['risk_factors'] = risk_factors
        if risk_score < 30:
            results['risk_level'] = 'LOW'; results['risk_color'] = (0, 255, 0)
        elif risk_score < 60:
            results['risk_level'] = 'MODERATE'; results['risk_color'] = (0, 255, 255)
        else:
            results['risk_level'] = 'HIGH'; results['risk_color'] = (0, 0, 255)
        self.risk_scores.append(risk_score); self.frames_analyzed += 1
        return results

    def _calc_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        ba = a - b; bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        cosine = np.dot(ba, bc) / denom
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _assess_injury_risk(self, knee, hip, front_w, back_w):
        risks = []; score = 0
        thresh = self.config.batting_injury_thresholds
        if knee < thresh['knee_angle_min']:
            score += 40
            risks.append({'type': 'Knee Joint Stress', 'severity': 'High',
                          'reason': f'Knee angle {knee:.1f}° too acute', 'fix': 'Straighten legs, avoid deep crouch'})
        elif knee < thresh['knee_angle_low']:
            score += 20
            risks.append({'type': 'Knee Strain Risk', 'severity': 'Medium',
                          'reason': f'Knee angle {knee:.1f}° moderately flexed', 'fix': 'Work on stance height'})
        if hip < thresh['hip_angle_min']:
            score += 40
            risks.append({'type': 'Lower Back Stress', 'severity': 'High',
                          'reason': f'Hip angle {hip:.1f}° - excessive forward bend',
                          'fix': 'Maintain upright posture, engage core'})
        elif hip < thresh['hip_angle_low']:
            score += 20
            risks.append({'type': 'Lumbar Strain Risk', 'severity': 'Medium',
                          'reason': f'Hip angle {hip:.1f}° shows forward lean', 'fix': 'Core strengthening'})
        if front_w > thresh['weight_imbalance_high'] or back_w > thresh['weight_imbalance_high']:
            score += 30
            risks.append({'type': 'Weight Imbalance', 'severity': 'High',
                          'reason': f'Excessive weight on one foot ({front_w:.1f}% / {back_w:.1f}%)',
                          'fix': 'Work on balanced stance'})
        return min(score, 100), risks

    def generate_report(self):
        if not self.risk_scores:
            return None
        return {
            'activity_type': 'BATTING',
            'frames_analyzed': self.frames_analyzed,
            'average_risk_score': round(statistics.mean(self.risk_scores), 1),
            'risk_level': 'LOW' if statistics.mean(self.risk_scores) < 30 else
                          ('MODERATE' if statistics.mean(self.risk_scores) < 60 else 'HIGH'),
            'average_knee_angle': round(statistics.mean(self.knee_angles), 1),
            'average_hip_angle': round(statistics.mean(self.hip_angles), 1),
            'average_front_weight': round(statistics.mean(self.front_weight_history), 1),
            'average_back_weight': round(statistics.mean(self.back_weight_history), 1),
            'weight_balance': round(abs(statistics.mean(self.front_weight_history) -
                                        statistics.mean(self.back_weight_history)), 1)
        }


class UnifiedCricketAnalyzer:
    def __init__(self, config=None):
        self.config = config or CricketAnalysisConfig()
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        self.activity_type = None
        self.landmarks_history = []
        self.batting_analyzer = None
        self.frame_idx = 0

    def analyze_video(self, video_path, output_dir, progress_callback=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = os.path.join(output_dir, "annotated_cricket.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

        detection_frames = 60

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(int((self.frame_idx / total_frames) * 85))

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(image_rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                landmarks = self._extract_landmarks(lm)
                if self.frame_idx <= detection_frames:
                    self.landmarks_history.append(landmarks)
                if self.frame_idx == detection_frames:
                    self.activity_type = ActivityDetector.detect_activity(self.landmarks_history)
                    if self.activity_type == 'BATTING':
                        self.batting_analyzer = BattingAnalyzer(self.config)
                if self.activity_type == 'BATTING' and self.batting_analyzer:
                    results = self.batting_analyzer.analyze_frame(landmarks, frame_w, frame_h)
                    frame = self._draw_batting(frame, results, res, frame_w, frame_h)
                else:
                    self.mp_draw.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    if self.frame_idx < detection_frames:
                        cv2.putText(frame, "Detecting activity...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            out.write(frame)

        cap.release(); out.release(); self.pose.close()

        report = self.batting_analyzer.generate_report() if self.batting_analyzer else None
        graphs = self._generate_plots(output_dir)
        self._save_reports(output_dir, report)

        if progress_callback:
            progress_callback(100)

        return {
            'video_path': output_video,
            'video_url': 'annotated_cricket.mp4',
            'activity_type': self.activity_type or 'UNKNOWN',
            'report': report,
            'graphs': graphs
        }

    def _extract_landmarks(self, lm):
        mp_pose = self.mp_pose.PoseLandmark
        landmarks = {
            'l_ankle': lm[mp_pose.LEFT_ANKLE], 'r_ankle': lm[mp_pose.RIGHT_ANKLE],
            'l_hip': lm[mp_pose.LEFT_HIP], 'r_hip': lm[mp_pose.RIGHT_HIP],
            'l_shoulder': lm[mp_pose.LEFT_SHOULDER], 'r_shoulder': lm[mp_pose.RIGHT_SHOULDER],
            'r_wrist': lm[mp_pose.RIGHT_WRIST], 'l_knee': lm[mp_pose.LEFT_KNEE],
            'r_knee': lm[mp_pose.RIGHT_KNEE],
            'hip_x': (lm[mp_pose.LEFT_HIP].x + lm[mp_pose.RIGHT_HIP].x) / 2
        }
        try:
            landmarks['l_foot'] = lm[mp_pose.LEFT_FOOT_INDEX]
            landmarks['r_foot'] = lm[mp_pose.RIGHT_FOOT_INDEX]
        except:
            pass
        return landmarks

    def _draw_batting(self, frame, results, res, frame_w, frame_h):
        self.mp_draw.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f'Front: {results["front_weight"]:.1f}%', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(frame, f'Back: {results["back_weight"]:.1f}%', (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
        cv2.putText(frame, f'Knee: {results["knee_angle"]:.1f}°', (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        cv2.putText(frame, f'Hip: {results["hip_angle"]:.1f}°', (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        cv2.putText(frame, f'Risk: {results["risk_level"]} ({results["risk_score"]:.0f}%)',
                    (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, results['risk_color'], 3)
        bar_y = frame_h - 60; bar_h = 35; bar_w = frame_w - 60; start_x = 30
        fw = int(bar_w * (results["front_weight"] / 100))
        cv2.rectangle(frame, (start_x, bar_y), (start_x + fw, bar_y + bar_h), (0, 200, 255), -1)
        cv2.rectangle(frame, (start_x + fw, bar_y), (start_x + bar_w, bar_y + bar_h), (0, 100, 255), -1)
        cv2.rectangle(frame, (start_x, bar_y), (start_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        return frame

    def _generate_plots(self, output_dir):
        graphs = []
        if not self.batting_analyzer:
            return graphs
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(self.batting_analyzer.front_weight_history, label="Front Foot %", color="#00ffaa", linewidth=2)
            plt.plot(self.batting_analyzer.back_weight_history, label="Back Foot %", color="#ff6600", linewidth=2)
            plt.xlabel("Frame"); plt.ylabel("Weight (%)"); plt.title("Weight Transfer - Batting")
            plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 100); plt.tight_layout()
            p = os.path.join(output_dir, "weight_transfer.png")
            plt.savefig(p, dpi=150); plt.close()
            graphs.append({'name': 'Weight Transfer', 'url': 'weight_transfer.png'})

            plt.figure(figsize=(12, 5))
            plt.plot(self.batting_analyzer.risk_scores, color='#ff0055', linewidth=2)
            plt.axhline(30, color='green', linestyle='--', alpha=0.6, label='Low/Moderate boundary')
            plt.axhline(60, color='red', linestyle='--', alpha=0.6, label='Moderate/High boundary')
            plt.xlabel("Frame"); plt.ylabel("Risk Score"); plt.title("Injury Risk Over Time")
            plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 100); plt.tight_layout()
            p = os.path.join(output_dir, "risk_over_time.png")
            plt.savefig(p, dpi=150); plt.close()
            graphs.append({'name': 'Injury Risk Over Time', 'url': 'risk_over_time.png'})

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(self.batting_analyzer.knee_angles, color='#00ccff', linewidth=2)
            ax1.axhline(100, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(120, color='orange', linestyle='--', alpha=0.5)
            ax1.set_ylabel("Knee Angle (°)"); ax1.set_title("Knee Angle"); ax1.grid(alpha=0.3)
            ax2.plot(self.batting_analyzer.hip_angles, color='#cc00ff', linewidth=2)
            ax2.axhline(110, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(130, color='orange', linestyle='--', alpha=0.5)
            ax2.set_xlabel("Frame"); ax2.set_ylabel("Hip Angle (°)"); ax2.set_title("Hip Angle"); ax2.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(output_dir, "joint_angles.png")
            plt.savefig(p, dpi=150); plt.close()
            graphs.append({'name': 'Joint Angles', 'url': 'joint_angles.png'})
        except Exception as e:
            print(f"Plot error: {e}")
        return graphs

    def _save_reports(self, output_dir, report):
        if not report:
            return
        try:
            with open(os.path.join(output_dir, "batting_analysis.json"), 'w') as f:
                json.dump(report, f, indent=2)
            with open(os.path.join(output_dir, "batting_summary.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for k, v in report.items():
                    writer.writerow([k, v])
        except Exception as e:
            print(f"Report save error: {e}")


class BattingAnalysisModule:
    def analyze(self, video_path, output_dir, progress_callback=None):
        config = CricketAnalysisConfig()
        analyzer = UnifiedCricketAnalyzer(config)
        return analyzer.analyze_video(video_path, output_dir, progress_callback)

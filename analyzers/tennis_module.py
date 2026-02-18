#!/usr/bin/env python3
"""
Tennis Injury Prediction Module
Adapted from tennis.py - converted from script to class-based module
Added: video path as parameter, output_dir configuration, progress callback
All analysis algorithms preserved exactly as original
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import json
from collections import deque


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_angular_velocity(angle_history, fps):
    if len(angle_history) < 2:
        return 0
    return (angle_history[-1] - angle_history[-2]) * fps


def calculate_trunk_tilt(shoulder_left, shoulder_right, hip_left, hip_right):
    shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
    shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
    hip_mid_x = (hip_left.x + hip_right.x) / 2
    hip_mid_y = (hip_left.y + hip_right.y) / 2
    dx = shoulder_mid_x - hip_mid_x; dy = shoulder_mid_y - hip_mid_y
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def calculate_hip_shoulder_separation(shoulder_left, shoulder_right, hip_left, hip_right):
    shoulder_angle = np.degrees(np.arctan2(shoulder_right.y - shoulder_left.y,
                                            shoulder_right.x - shoulder_left.x))
    hip_angle = np.degrees(np.arctan2(hip_right.y - hip_left.y, hip_right.x - hip_left.x))
    return abs(shoulder_angle - hip_angle)


def create_mini_graph(data, title, color, max_len=100, y_min=0, y_max=180):
    fig, ax = plt.subplots(figsize=(3, 1.5), dpi=80)
    if len(data) > 0:
        x = list(range(len(data)))
        ax.plot(x, data, color=color, linewidth=2)
        ax.fill_between(x, data, alpha=0.3, color=color)
    ax.set_xlim(0, max_len); ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_facecolor('#1a1a1a'); fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=6)
    for spine in ax.spines.values(): spine.set_color('white')
    ax.title.set_color('white')
    canvas = FigureCanvasAgg(fig); canvas.draw()
    buf = canvas.buffer_rgba()
    graph_image = np.asarray(buf); plt.close(fig)
    return cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)


def overlay_graph(frame, graph_img, x, y):
    h, w = graph_img.shape[:2]
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame
    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.1, graph_img, 0.9, 0)
    return frame


def calculate_injury_risk(metrics):
    risk_score = 0; risk_factors = []
    if metrics['elbow_angle'] < 140:
        risk_score += 15; risk_factors.append("Excessive elbow flexion")
    if metrics['elbow_velocity'] > 800:
        risk_score += 20; risk_factors.append("High elbow extension speed")
    if metrics['shoulder_angle'] > 160:
        risk_score += 15; risk_factors.append("Excessive shoulder abduction")
    if metrics['shoulder_velocity'] > 600:
        risk_score += 15; risk_factors.append("High shoulder rotation speed")
    if metrics['trunk_tilt'] > 25:
        risk_score += 20; risk_factors.append("Excessive trunk lean")
    if metrics['knee_angle'] < 100:
        risk_score += 15; risk_factors.append("Deep knee flexion")
    if metrics['hip_shoulder_sep'] > 50:
        risk_score += 15; risk_factors.append("Excessive hip-shoulder separation")
    return min(risk_score, 100), risk_factors


class TennisBiomechanicsAnalyzer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def analyze_video(self, video_path, output_dir, progress_callback=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        output_video = os.path.join(output_dir, "annotated_tennis.mp4")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        MAX_HISTORY = 100
        detected_factors = {}   # track how many frames each risk factor fires
        shoulder_angles = deque(maxlen=MAX_HISTORY)
        elbow_angles = deque(maxlen=MAX_HISTORY)
        knee_angles = deque(maxlen=MAX_HISTORY)
        trunk_tilts = deque(maxlen=MAX_HISTORY)
        hip_shoulder_seps = deque(maxlen=MAX_HISTORY)
        shoulder_velocities = deque(maxlen=MAX_HISTORY)
        elbow_velocities = deque(maxlen=MAX_HISTORY)
        injury_risks = deque(maxlen=MAX_HISTORY)
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(int((frame_count / total_frames) * 80))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                le = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                re = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                lk = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                rk = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                la = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

                shoulder_angle = calculate_angle(rs, re, rw)
                shoulder_angles.append(shoulder_angle)
                elbow_angle = calculate_angle(rs, re, rw)
                elbow_angles.append(elbow_angle)
                left_knee_angle = calculate_angle(lh, lk, la)
                right_knee_angle = calculate_angle(rh, rk, ra)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
                knee_angles.append(avg_knee_angle)
                trunk_tilt = calculate_trunk_tilt(ls, rs, lh, rh)
                trunk_tilts.append(trunk_tilt)
                hip_shoulder_sep = calculate_hip_shoulder_separation(ls, rs, lh, rh)
                hip_shoulder_seps.append(hip_shoulder_sep)
                shoulder_vel = calculate_angular_velocity(list(shoulder_angles), fps)
                elbow_vel = calculate_angular_velocity(list(elbow_angles), fps)
                shoulder_velocities.append(abs(shoulder_vel))
                elbow_velocities.append(abs(elbow_vel))

                metrics = {
                    'shoulder_angle': shoulder_angle, 'elbow_angle': elbow_angle,
                    'knee_angle': avg_knee_angle, 'trunk_tilt': trunk_tilt,
                    'hip_shoulder_sep': hip_shoulder_sep,
                    'shoulder_velocity': abs(shoulder_vel), 'elbow_velocity': abs(elbow_vel)
                }
                risk_score, risk_factors = calculate_injury_risk(metrics)
                injury_risks.append(risk_score)
                for rf in risk_factors:
                    detected_factors[rf] = detected_factors.get(rf, 0) + 1

                if risk_score < 30:
                    risk_label = "LOW RISK"; risk_color = (0, 255, 0); bar_color = (0, 255, 0)
                elif risk_score < 60:
                    risk_label = "MODERATE RISK"; risk_color = (0, 255, 255); bar_color = (0, 255, 255)
                elif risk_score < 80:
                    risk_label = "HIGH RISK"; risk_color = (0, 165, 255); bar_color = (0, 165, 255)
                else:
                    risk_label = "CRITICAL RISK"; risk_color = (0, 0, 255); bar_color = (0, 0, 255)

                self.mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

                # Mini graphs
                graph1 = create_mini_graph(list(shoulder_angles), 'Shoulder Angle (°)', '#00ff00', y_min=0, y_max=180)
                graph2 = create_mini_graph(list(elbow_angles), 'Elbow Angle (°)', '#ffff00', y_min=0, y_max=180)
                graph3 = create_mini_graph(list(elbow_velocities), 'Elbow Velocity (°/s)', '#ff00ff', y_min=0, y_max=1200)
                graph4 = create_mini_graph(list(trunk_tilts), 'Trunk Tilt (°)', '#00ffff', y_min=0, y_max=50)
                graph5 = create_mini_graph(list(injury_risks), 'Injury Risk (%)', '#ff0000', y_min=0, y_max=100)
                graph6 = create_mini_graph(list(knee_angles), 'Knee Flexion (°)', '#ffa500', y_min=0, y_max=180)

                graph_x = width - 260; graph_y_start = 20; graph_spacing = 130
                frame = overlay_graph(frame, graph1, graph_x, graph_y_start)
                frame = overlay_graph(frame, graph2, graph_x, graph_y_start + graph_spacing)
                frame = overlay_graph(frame, graph3, graph_x, graph_y_start + graph_spacing * 2)
                frame = overlay_graph(frame, graph4, graph_x, graph_y_start + graph_spacing * 3)
                frame = overlay_graph(frame, graph5, graph_x, graph_y_start + graph_spacing * 4)
                frame = overlay_graph(frame, graph6, 10, graph_y_start + graph_spacing * 4)

                y_offset = 30
                cv2.putText(frame, f'Shoulder: {shoulder_angle:.1f}°', (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35
                cv2.putText(frame, f'Elbow: {elbow_angle:.1f}°', (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 35
                cv2.putText(frame, f'Knee: {avg_knee_angle:.1f}°', (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                y_offset += 35
                cv2.putText(frame, f'Trunk Tilt: {trunk_tilt:.1f}°', (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 35
                cv2.putText(frame, f'Elbow Vel: {abs(elbow_vel):.0f}°/s', (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                bar_height = 40; bar_y = height - 100; bar_width = width - 60; bar_x = 30
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                risk_bar_width = int(bar_width * (risk_score / 100))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + risk_bar_width, bar_y + bar_height), bar_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 3)
                cv2.putText(frame, f'{risk_score:.0f}%', (bar_x + bar_width//2 - 40, bar_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                label_y = height - 50
                cv2.putText(frame, f'INJURY RISK: {risk_label}', (bar_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)
                if risk_factors:
                    cv2.putText(frame, f'! {risk_factors[0]}', (bar_x, label_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

        cap.release(); out.release(); self.pose.close()

        graphs = []
        avg_risk = float(np.mean(list(injury_risks))) if injury_risks else 0
        max_risk = float(np.max(list(injury_risks))) if injury_risks else 0

        if avg_risk < 30:
            overall_risk = "LOW"
        elif avg_risk < 60:
            overall_risk = "MODERATE"
        else:
            overall_risk = "HIGH"

        # Summary graph
        try:
            if len(injury_risks) > 0:
                fig, axes = plt.subplots(3, 2, figsize=(14, 10))
                fig.suptitle('Tennis Biomechanics & Injury Risk Analysis', fontsize=16, fontweight='bold')

                axes[0, 0].plot(list(shoulder_angles), color='green', linewidth=2)
                axes[0, 0].set_title('Shoulder Angle Over Time'); axes[0, 0].set_ylabel('Angle (degrees)')
                axes[0, 0].grid(True, alpha=0.3); axes[0, 0].axhline(y=160, color='r', linestyle='--', label='Risk Threshold')
                axes[0, 0].legend()

                axes[0, 1].plot(list(elbow_angles), color='yellow', linewidth=2)
                axes[0, 1].set_title('Elbow Angle Over Time'); axes[0, 1].set_ylabel('Angle (degrees)')
                axes[0, 1].grid(True, alpha=0.3); axes[0, 1].axhline(y=140, color='r', linestyle='--', label='Risk Threshold')
                axes[0, 1].legend()

                axes[1, 0].plot(list(elbow_velocities), color='magenta', linewidth=2)
                axes[1, 0].set_title('Elbow Angular Velocity'); axes[1, 0].set_ylabel('Velocity (deg/sec)')
                axes[1, 0].grid(True, alpha=0.3); axes[1, 0].axhline(y=800, color='r', linestyle='--', label='Risk Threshold')
                axes[1, 0].legend()

                axes[1, 1].plot(list(trunk_tilts), color='cyan', linewidth=2)
                axes[1, 1].set_title('Trunk Tilt (Lateral Lean)'); axes[1, 1].set_ylabel('Angle (degrees)')
                axes[1, 1].grid(True, alpha=0.3); axes[1, 1].axhline(y=25, color='r', linestyle='--', label='Risk Threshold')
                axes[1, 1].legend()

                axes[2, 0].plot(list(knee_angles), color='orange', linewidth=2)
                axes[2, 0].set_title('Knee Flexion Angle'); axes[2, 0].set_ylabel('Angle (degrees)')
                axes[2, 0].set_xlabel('Frame'); axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].axhline(y=100, color='r', linestyle='--', label='Risk Threshold'); axes[2, 0].legend()

                injury_list = list(injury_risks)
                axes[2, 1].plot(injury_list, color='red', linewidth=2)
                axes[2, 1].fill_between(range(len(injury_list)), injury_list, alpha=0.3, color='red')
                axes[2, 1].set_title('Overall Injury Risk Score'); axes[2, 1].set_ylabel('Risk (%)')
                axes[2, 1].set_xlabel('Frame'); axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].axhline(y=60, color='orange', linestyle='--', label='High Risk')
                axes[2, 1].axhline(y=30, color='yellow', linestyle='--', label='Moderate Risk')
                axes[2, 1].legend()

                plt.tight_layout()
                p = os.path.join(output_dir, "tennis_analysis_summary.png")
                plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
                graphs.append({'name': 'Tennis Biomechanics Summary', 'url': 'tennis_analysis_summary.png'})

                # Individual graphs
                plt.figure(figsize=(12, 4))
                plt.plot(list(injury_risks), color='#ff0055', linewidth=2, label='Injury Risk %')
                plt.fill_between(range(len(list(injury_risks))), list(injury_risks), alpha=0.3, color='#ff0055')
                plt.axhline(60, color='orange', linestyle='--'); plt.axhline(30, color='yellow', linestyle='--')
                plt.xlabel("Frame"); plt.ylabel("Risk %"); plt.title("Injury Risk Score Over Time")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                p = os.path.join(output_dir, "injury_risk_timeline.png")
                plt.savefig(p, dpi=150); plt.close()
                graphs.append({'name': 'Injury Risk Timeline', 'url': 'injury_risk_timeline.png'})

        except Exception as e:
            print(f"Plot error: {e}")

        summary = {
            'total_frames': frame_count,
            'average_risk': round(avg_risk, 1),
            'max_risk': round(max_risk, 1),
            'overall_risk': overall_risk
        }

        suggestions = self._generate_suggestions(detected_factors)

        if progress_callback:
            progress_callback(100)

        return {
            'video_path': output_video,
            'video_url': 'annotated_tennis.mp4',
            'summary': summary,
            'suggestions': suggestions,
            'graphs': graphs
        }


    def _generate_suggestions(self, detected_factors):
        """
        Generate specific, actionable suggestions based on detected tennis risk factors.
        detected_factors: dict of {factor_string: frame_count}
        """
        SUGGESTION_MAP = {
            'Excessive elbow flexion': {
                'icon': '💪',
                'injury': 'Tennis Elbow Risk (Medial Epicondylitis)',
                'severity': 'High',
                'reason': 'Elbow angle below 140° during stroke — the joint is highly flexed, increasing tendon load',
                'suggestions': [
                    'Ensure the elbow is at least partially extended before contact — avoid "chicken winging" the arm',
                    'Eccentric wrist extensor exercises: Tyler Twist with a Flexbar (3×15 daily) — clinically proven for tennis elbow',
                    'Check grip size — an undersized grip causes the forearm muscles to overwork and flex the elbow excessively',
                    'Practice shadow swings focusing on a straighter arm at the point of contact',
                    'Apply forearm strap (counterforce brace) during play to reduce tendon load',
                ]
            },
            'High elbow extension speed': {
                'icon': '⚡',
                'injury': 'UCL / Medial Elbow Ligament Stress',
                'severity': 'High',
                'reason': 'Rapid elbow snap-through (>800°/s) places high valgus stress on the medial elbow ligament',
                'suggestions': [
                    'Slow down arm acceleration during practice — bowl at 70% pace to re-groove a smoother arc',
                    'Focus on generating power from the legs and trunk rotation rather than the arm snap',
                    'Wrist flexor strengthening: dumbbell wrist curls and reverse curls (3×15, light weight)',
                    'Ice the medial elbow for 15 min after every session involving heavy hitting',
                    'Use a slightly heavier racket to dampen vibration and reduce the snap-through reflex',
                ]
            },
            'Excessive shoulder abduction': {
                'icon': '🦴',
                'injury': 'Shoulder Impingement Syndrome',
                'severity': 'High',
                'reason': 'Shoulder angle above 160° indicates the arm is being raised beyond a safe range, compressing the subacromial space',
                'suggestions': [
                    'Lower your service toss slightly — a ball tossed too high forces the shoulder into extreme abduction',
                    'Rotator cuff strengthening: side-lying external rotation, prone Y-T-W raises (3×15 each)',
                    'Scapular stabilisation: band pull-aparts and wall slides (3×20)',
                    'Avoid overhead serving or smashing when shoulder is fatigued — most impingement injuries occur late in matches',
                    'Have a coach check your trophy position — the elbow should not be above shoulder height at ball release',
                ]
            },
            'High shoulder rotation speed': {
                'icon': '🔄',
                'injury': 'Rotator Cuff Strain / SLAP Tear Risk',
                'severity': 'High',
                'reason': 'Shoulder angular velocity >600°/s — the rotator cuff muscles must decelerate this forcefully, risking micro-tears',
                'suggestions': [
                    'Incorporate posterior shoulder and rotator cuff deceleration training: prone Y-T-W, face pulls (3×15)',
                    'Ensure proper follow-through — a curtailed follow-through forces the rotator cuff to do more braking',
                    'Reduce serve/overhead volume in training by 20–25% and build back gradually',
                    'Posterior capsule stretching: sleeper stretch and cross-body stretch (30s each, 3 sets daily)',
                    'Warm up rotationally: arm circles, band internal/external rotation before hitting',
                ]
            },
            'Excessive trunk lean': {
                'icon': '⚠️',
                'injury': 'Lumbar Spine / Lower Back Strain',
                'severity': 'Medium',
                'reason': 'Trunk tilt above 25° — excessive lateral lean compresses the facet joints and stresses the lumbar erectors',
                'suggestions': [
                    'Focus on staying tall and balanced through the shot — lean in from the hips, not by side-bending the spine',
                    'Core anti-lateral flexion strengthening: suitcase carries, side planks (3×45s each side)',
                    'Hip mobility work: 90/90 stretches, pigeon pose — tight hips force the trunk to compensate',
                    'Check your ready position stance — feet too narrow causes body lean on wide balls',
                    'McGill Big Three rehab exercises: bird-dog, curl-up, side plank if back pain is present',
                ]
            },
            'Deep knee flexion': {
                'icon': '🦵',
                'injury': 'Patellar Tendinopathy / Knee Cartilage Stress',
                'severity': 'Medium',
                'reason': 'Knee angle below 100° — deep bends significantly increase patellofemoral compression and tendon load',
                'suggestions': [
                    'Maintain a higher athletic stance — aim to keep knees between 110–140° during ready position and split step',
                    'Eccentric quad strengthening: decline-board slow squats (3×15) — gold standard for patellar tendinopathy',
                    'Avoid playing on very low-bounce surfaces (clay) with a deep crouch for extended periods without acclimatisation',
                    'VMO (inner quad) activation: terminal knee extensions with a resistance band (3×20)',
                    'Knee sleeve or patellar tendon strap during high-load training sessions',
                ]
            },
            'Excessive hip-shoulder separation': {
                'icon': '↔️',
                'injury': 'Oblique / Core Rotational Strain',
                'severity': 'Medium',
                'reason': 'Hip-shoulder separation above 50° — the torso is over-rotating relative to the hips, straining the obliques and thoracic spine',
                'suggestions': [
                    'Work on timing: let hips initiate the rotation first, with shoulders following — do not force extra separation',
                    'Thoracic rotation drills: seated rotations, open-book stretches (10 reps each side daily)',
                    'Anti-rotation core: Pallof press, half-kneeling cable chops (3×12 each side)',
                    'Oblique strengthening: cable woodchops, medicine ball rotational throws (3×10 each side)',
                    'During practice, focus on controlled rotation finish — avoid twisting the torso beyond the natural endpoint',
                ]
            },
        }

        # Determine severity based on how often each factor fired
        total_frames = max(sum(detected_factors.values()), 1) if detected_factors else 1

        # Sort by frequency (most persistent risks first)
        sorted_factors = sorted(detected_factors.items(), key=lambda x: -x[1])

        result = []
        for factor, count in sorted_factors:
            if factor not in SUGGESTION_MAP:
                continue
            info = SUGGESTION_MAP[factor]
            freq_pct = round((count / total_frames) * 100, 1)
            # Escalate severity if the factor fires in >50% of frames
            severity = info['severity']
            if freq_pct > 50 and severity == 'Medium':
                severity = 'High'

            result.append({
                'injury': info['injury'],
                'factor': factor,
                'severity': severity,
                'phase': 'Stroke Mechanics',
                'icon': info['icon'],
                'reasons': [info['reason'], f'Detected in {freq_pct}% of analysed frames ({count} frames)'],
                'suggestions': info['suggestions'],
                'occurrences': count
            })

        return result


class TennisAnalysisModule:
    def analyze(self, video_path, output_dir, progress_callback=None):
        analyzer = TennisBiomechanicsAnalyzer()
        return analyzer.analyze_video(video_path, output_dir, progress_callback)

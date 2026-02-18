#!/usr/bin/env python3
"""
Bowling Biomechanics Analysis Module
Adapted for web app use - no GUI picker, accepts output_dir as parameter
Original: bowling_analyzer.py (all analysis logic preserved)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
import shutil
from collections import deque
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


class BowlingAnalysisConfig:
    def __init__(self, output_dir="outputs"):
        self.calibration_seconds = 2.5
        self.delivery_window_frames = 8
        self.output_dir = output_dir
        self.save_plots = True
        self.save_json = True
        self.save_csv = True
        self.min_detection_confidence = 0.6
        self.min_tracking_confidence = 0.6
        self.injury_thresholds = {
            'elbow_hyperextension': 15.0,
            'knee_flexion_min': 140.0,
            'back_flexion_high_risk': 130.0,
            'shoulder_rotation_high': 170.0,
            'hip_rotation_excessive': 80.0,
            'landing_impact_high': 1.5,
            'run_up_asymmetry_high': 0.3
        }


class InjuryRiskAnalyzer:
    def __init__(self, config):
        self.config = config
        self.phase_risks = {
            'RUN-UP': [], 'GATHER': [], 'JUMP': [], 'PLANT': [],
            'DELIVERY_STRIDE': [], 'BALL_RELEASE': [], 'FOLLOW_THROUGH': []
        }

    def assess_runup_phase(self, hip_velocity, ankle_symmetry, stride_length_var):
        risks = []
        score = 0
        if hip_velocity > 0.15:
            score += 15
            risks.append({'type': 'Hamstring Strain', 'severity': 'Medium',
                          'reason': 'High hip velocity during run-up increases hamstring load'})
        if ankle_symmetry > self.config.injury_thresholds['run_up_asymmetry_high']:
            score += 20
            risks.append({'type': 'Unilateral Hip/Knee Stress', 'severity': 'High',
                          'reason': 'Asymmetric loading pattern detected in run-up'})
        if stride_length_var > 0.25:
            score += 10
            risks.append({'type': 'Coordination Issues', 'severity': 'Low',
                          'reason': 'Inconsistent stride length may indicate fatigue or poor mechanics'})
        return min(score, 100), risks

    def assess_jump_phase(self, jump_height, takeoff_velocity, ankle_angles):
        risks = []
        score = 0
        if jump_height > 0.08:
            score += 15
            risks.append({'type': 'Patellar Tendinopathy', 'severity': 'Medium',
                          'reason': 'High jump loads the knee extensors significantly'})
        if takeoff_velocity > 0.12:
            score += 12
            risks.append({'type': 'Achilles Tendon Stress', 'severity': 'Medium',
                          'reason': 'Rapid plantar flexion increases Achilles load'})
        if ankle_angles.get('inversion', 0) > 15:
            score += 18
            risks.append({'type': 'Ankle Sprain', 'severity': 'High',
                          'reason': 'Excessive ankle inversion during takeoff'})
        return min(score, 100), risks

    def assess_landing_phase(self, landing_velocity, knee_flexion, hip_drop):
        risks = []
        score = 0
        if landing_velocity > self.config.injury_thresholds['landing_impact_high']:
            score += 25
            risks.append({'type': 'Stress Fracture (Lumbar)', 'severity': 'High',
                          'reason': 'High impact forces during landing transmitted to spine'})
            risks.append({'type': 'Knee Cartilage Damage', 'severity': 'High',
                          'reason': 'Hard landing increases compressive forces on knee joint'})
        if knee_flexion < self.config.injury_thresholds['knee_flexion_min']:
            score += 30
            risks.append({'type': 'Lumbar Stress Fracture', 'severity': 'Critical',
                          'reason': 'Stiff-legged landing transfers forces directly to spine'})
        if hip_drop > 0.05:
            score += 15
            risks.append({'type': 'Hip Abductor Weakness', 'severity': 'Medium',
                          'reason': 'Contralateral hip drop indicates gluteal weakness'})
        return min(score, 100), risks

    def assess_delivery_phase(self, elbow_extension, back_angle, shoulder_rotation,
                               shoulder_alignment, hip_rotation):
        risks = []
        score = 0
        if elbow_extension > self.config.injury_thresholds['elbow_hyperextension']:
            score += 35
            risks.append({'type': 'Elbow Ligament Damage', 'severity': 'Critical',
                          'reason': f'Elbow extension of {elbow_extension:.1f}° exceeds safe limit (15°)'})
        elif elbow_extension > 12:
            score += 20
            risks.append({'type': 'Elbow Stress', 'severity': 'High',
                          'reason': f'Elbow extension of {elbow_extension:.1f}° approaching danger zone'})
        if back_angle < self.config.injury_thresholds['back_flexion_high_risk']:
            score += 30
            risks.append({'type': 'Lumbar Disc Injury', 'severity': 'Critical',
                          'reason': 'Excessive lateral flexion with rotation (mixed action)'})
        elif back_angle < 140:
            score += 20
            risks.append({'type': 'Lower Back Strain', 'severity': 'High',
                          'reason': 'Significant spinal flexion during delivery'})
        if shoulder_rotation > self.config.injury_thresholds['shoulder_rotation_high']:
            score += 18
            risks.append({'type': 'Shoulder Impingement', 'severity': 'Medium',
                          'reason': 'Excessive shoulder circumduction increases impingement risk'})
        if shoulder_alignment == "FRONT-ON" and hip_rotation > self.config.injury_thresholds['hip_rotation_excessive']:
            score += 25
            risks.append({'type': 'Shoulder Instability', 'severity': 'High',
                          'reason': 'Front-on action with hip rotation creates shoulder counter-rotation stress'})
        if shoulder_rotation > 150:
            score += 15
            risks.append({'type': 'Rotator Cuff Strain', 'severity': 'Medium',
                          'reason': 'High shoulder external rotation loads rotator cuff'})
        return min(score, 100), risks

    def assess_followthrough_phase(self, deceleration_rate, shoulder_load, trunk_rotation):
        risks = []
        score = 0
        if deceleration_rate > 0.18:
            score += 15
            risks.append({'type': 'Posterior Shoulder Stress', 'severity': 'Medium',
                          'reason': 'Rapid arm deceleration loads posterior shoulder structures'})
        if shoulder_load > 0.15:
            score += 20
            risks.append({'type': 'Labral Tear Risk', 'severity': 'High',
                          'reason': 'Abrupt stopping of arm increases risk of labral injury'})
        if trunk_rotation > 70:
            score += 12
            risks.append({'type': 'Oblique Strain', 'severity': 'Medium',
                          'reason': 'Excessive trunk rotation during follow-through'})
        return min(score, 100), risks

    def generate_injury_report(self):
        total_score = 0
        all_risks = []
        phase_scores = {}
        for phase, risk_list in self.phase_risks.items():
            if risk_list:
                phase_score = sum(r['score'] for r in risk_list) / len(risk_list)
                phase_scores[phase] = phase_score
                total_score += phase_score
                all_risks.extend(risk_list)
        avg_score = total_score / max(len(phase_scores), 1) if phase_scores else 0
        if avg_score < 30:
            risk_level, risk_color = "LOW", "GREEN"
        elif avg_score < 60:
            risk_level, risk_color = "MODERATE", "YELLOW"
        else:
            risk_level, risk_color = "HIGH", "RED"
        return {
            'overall_score': round(avg_score, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'phase_scores': phase_scores,
            'all_risks': all_risks
        }

    def generate_suggestions(self, all_risks):
        """
        Generate specific, actionable suggestions based on detected injury risks.
        Each suggestion is tied to an actual injury type found in the video analysis.
        """
        SUGGESTION_MAP = {
            'Hamstring Strain': {
                'icon': '🦵',
                'phase': 'Run-Up',
                'suggestions': [
                    'Reduce run-up pace by 10–15% and build back up gradually over 2–3 weeks',
                    'Add Nordic hamstring curls (3×6) to your weekly strength programme',
                    'Perform dynamic leg swings and hip flexor stretches before every session',
                    'Focus on hip-drive mechanics — drive knees forward rather than pushing off hard with the toes',
                ]
            },
            'Unilateral Hip/Knee Stress': {
                'icon': '⚖️',
                'phase': 'Run-Up',
                'suggestions': [
                    'Video your run-up from behind: both feet should land on or very close to the same line',
                    'Practice single-leg balance drills (30s each leg, 3 sets) to correct strength imbalance',
                    'Use a straight run-up chalk line on the pitch during nets to enforce symmetry',
                    'Check footwear — worn-out soles on one side cause asymmetric loading',
                ]
            },
            'Coordination Issues': {
                'icon': '🎯',
                'phase': 'Run-Up',
                'suggestions': [
                    'Mark your run-up with cones at fixed intervals and rehearse it at walk pace first',
                    'Record and count strides each delivery — inconsistency of more than 1 stride needs correction',
                    'Work with a coach on a shorter, more controlled 8–10 step run-up if fatigue is a factor',
                ]
            },
            'Patellar Tendinopathy': {
                'icon': '🦴',
                'phase': 'Jump / Gather',
                'suggestions': [
                    'Replace high-jump bound with a lower, more controlled gather step — aim for minimal air time',
                    'Add eccentric quad strengthening: slow decline-board squats (3×15) 3× per week',
                    'Apply ice to kneecap area for 15 min immediately after bowling sessions',
                    'Avoid bowling on hard surfaces (concrete/astroturf) without adequate warm-up',
                ]
            },
            'Achilles Tendon Stress': {
                'icon': '🦶',
                'phase': 'Jump / Gather',
                'suggestions': [
                    'Perform calf raises with slow 3-second lowering phase daily (3×15 each leg)',
                    'Ensure you land mid-foot, not on your toes, during the gather phase',
                    'Replace hard explosive jumps with a smooth, controlled stride gather',
                    'Check bowling boot heel height — inadequate cushioning increases Achilles load',
                ]
            },
            'Ankle Sprain': {
                'icon': '🩹',
                'phase': 'Jump / Gather',
                'suggestions': [
                    'Use ankle bracing or strapping tape during bowling sessions and matches',
                    'Perform wobble-board balance exercises daily (3×60s each ankle)',
                    'Check your gather — ensure foot lands flat, not on the lateral edge',
                    'Strengthen peroneal muscles: resistance-band eversion exercises (3×15 each side)',
                ]
            },
            'Stress Fracture (Lumbar)': {
                'icon': '🚨',
                'phase': 'Landing / Plant',
                'suggestions': [
                    'URGENT: Rest from bowling immediately and seek a physiotherapist or sports doctor assessment',
                    'Land with a bent knee (at least 140°) to absorb impact before it reaches the spine',
                    'Strengthen core: dead bugs, bird-dogs, and McGill curl-ups daily before returning',
                    'Avoid bowling more than 12–15 overs per day until technique is corrected',
                    'Work with a biomechanics coach specifically on your front-foot plant mechanics',
                ]
            },
            'Knee Cartilage Damage': {
                'icon': '⚠️',
                'phase': 'Landing / Plant',
                'suggestions': [
                    'Land with knee bent — stiff-legged landings compress cartilage directly',
                    'Strengthen VMO (inner quad): terminal knee extensions with a resistance band (3×20)',
                    'Avoid bowling on extremely hard surfaces without adequate cushioning in boots',
                    'Swimming and cycling are safe cross-training to maintain fitness while reducing knee load',
                ]
            },
            'Lumbar Stress Fracture': {
                'icon': '🚨',
                'phase': 'Landing / Plant',
                'suggestions': [
                    'URGENT: Stop bowling and get a CT or MRI scan — lumbar stress fractures can become career-ending',
                    'Transition to a side-on or semi-upright action to reduce spinal flexion at landing',
                    'Build posterior chain strength: Romanian deadlifts and glute bridges (3×12)',
                    'Practice landing with at least 30° knee bend as a rule — never land with a straight leg',
                ]
            },
            'Hip Abductor Weakness': {
                'icon': '🦵',
                'phase': 'Landing / Plant',
                'suggestions': [
                    'Add lateral band walks (3×20 steps each direction) to every warm-up',
                    'Side-lying hip abduction with resistance band: 3×15 each leg',
                    'Single-leg squats in front of a mirror — ensure the hip does not drop on the standing side',
                    'Clamshell exercises: 3×20 each side, 3× per week',
                ]
            },
            'Elbow Ligament Damage': {
                'icon': '🚨',
                'phase': 'Delivery',
                'suggestions': [
                    'URGENT: ICC rules require elbow extension ≤15° — have your action officially reviewed by your board',
                    'Work with a bowling coach immediately to retrain your release point and arm path',
                    'Wrist flexor and extensor strengthening: dumbbell wrist curls (light weight, high rep)',
                    'Record your delivery from directly in front at slow motion to visualise the straightening',
                    'Consider a temporary transition to off-spin or medium pace while retraining the action',
                ]
            },
            'Elbow Stress': {
                'icon': '💪',
                'phase': 'Delivery',
                'suggestions': [
                    'Elbow extension is approaching the ICC 15° limit — address technique now before it becomes illegal',
                    'Focus on a high, loose arm at release — a tense arm tends to snap through and extend',
                    'Add eccentric forearm curls and reverse wrist curls to strengthen the medial elbow',
                    'Bowl at 70% pace in nets focusing purely on arm path before returning to full pace',
                ]
            },
            'Lumbar Disc Injury': {
                'icon': '🚨',
                'phase': 'Delivery',
                'suggestions': [
                    'URGENT: Cease bowling and seek physiotherapy — mixed actions with lateral flexion cause disc injury',
                    'Transition to a pure side-on or front-on action — mixed actions create the most damaging spinal forces',
                    'McGill "Big Three" rehabilitation: bird-dog, side plank, McGill curl-up daily',
                    'Core anti-rotation exercises: Pallof press, half-kneeling chops (3×12)',
                    'Return to bowling should be supervised and gradual — no more than 3 overs in early return',
                ]
            },
            'Lower Back Strain': {
                'icon': '⚠️',
                'phase': 'Delivery',
                'suggestions': [
                    'Adopt a more consistent (less mixed) bowling action — side-on or front-on, not a combination',
                    'Hip mobility work: 90/90 hip stretches, pigeon pose, and hip flexor stretches daily',
                    'Strengthen spinal extensors: back extensions on a Roman chair (3×15)',
                    'Reduce bowling load by 30% for 2 weeks and monitor back symptoms',
                ]
            },
            'Shoulder Impingement': {
                'icon': '💪',
                'phase': 'Delivery',
                'suggestions': [
                    'Lower the release point slightly — a very high arm creates impingement at the acromion',
                    'Rotator cuff strengthening: internal/external rotation with resistance band (3×15 each)',
                    'Scapular stabilisation: wall slides, band pull-aparts (3×20)',
                    'Avoid bowling when the shoulder is fatigued — impingement worsens with fatigue',
                ]
            },
            'Shoulder Instability': {
                'icon': '⚠️',
                'phase': 'Delivery',
                'suggestions': [
                    'The front-on action with high hip rotation creates severe counter-rotation at the shoulder — consider transitioning to a side-on action',
                    'Rotator cuff and posterior shoulder strengthening: face pulls, Y-T-W raises (3×15)',
                    'Proprioception exercises: body-blade or perturbation training for shoulder stability',
                    'Tape or brace the shoulder during heavy bowling sessions until strength is restored',
                ]
            },
            'Rotator Cuff Strain': {
                'icon': '💪',
                'phase': 'Delivery',
                'suggestions': [
                    'Reduce bowling volume and pace by 25% for 2–3 weeks',
                    'Daily rotator cuff routine: side-lying external rotation, prone Y-T-W (3×15)',
                    'Ensure your arm reaches vertical before starting the delivery arc — early rotation loads the cuff',
                    'Apply heat before bowling and ice for 15 min after each session',
                ]
            },
            'Posterior Shoulder Stress': {
                'icon': '🔄',
                'phase': 'Follow-Through',
                'suggestions': [
                    'Allow the arm to decelerate naturally across the body — do not "brace" or stop it abruptly',
                    'Posterior capsule stretching: cross-body stretch, sleeper stretch (30s each, 3 sets)',
                    'Strengthen posterior shoulder: face pulls, rear-delt flies (3×15)',
                    'Ensure follow-through carries the bowling arm all the way through past the hip',
                ]
            },
            'Labral Tear Risk': {
                'icon': '🚨',
                'phase': 'Follow-Through',
                'suggestions': [
                    'URGENT: An abrupt follow-through stop is the primary cause of labral injuries — consult a sports physio',
                    'Consciously extend the follow-through: arm should finish pointing at the batsman or below',
                    'Avoid bowling through shoulder pain — the labrum does not heal well without rest',
                    'Posterior shoulder strengthening is essential: resistance band rows and face pulls daily',
                ]
            },
            'Oblique Strain': {
                'icon': '🔄',
                'phase': 'Follow-Through',
                'suggestions': [
                    'Reduce trunk rotation in follow-through by keeping hips and shoulders more aligned',
                    'Side plank progressions: static hold then hip dips (3×30s–45s each side)',
                    'Anti-rotation core work: Pallof press, cable woodchops (3×12)',
                    'Stretch obliques daily: standing lateral stretch, foam roller thoracic rotation',
                ]
            },
        }

        # Collect all unique injury types detected across all phases
        detected_types = {}
        for risk_entry in all_risks:
            for factor in risk_entry.get('factors', []):
                injury_type = factor['type']
                severity = factor['severity']
                reason = factor['reason']
                if injury_type not in detected_types:
                    detected_types[injury_type] = {
                        'severity': severity,
                        'reasons': [],
                        'count': 0
                    }
                if reason not in detected_types[injury_type]['reasons']:
                    detected_types[injury_type]['reasons'].append(reason)
                detected_types[injury_type]['count'] += 1

        # Sort by severity
        severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        sorted_injuries = sorted(
            detected_types.items(),
            key=lambda x: severity_order.get(x[1]['severity'], 4)
        )

        result = []
        for injury_type, data in sorted_injuries:
            if injury_type in SUGGESTION_MAP:
                info = SUGGESTION_MAP[injury_type]
                result.append({
                    'injury': injury_type,
                    'severity': data['severity'],
                    'phase': info['phase'],
                    'icon': info['icon'],
                    'reasons': data['reasons'],
                    'suggestions': info['suggestions'],
                    'occurrences': data['count']
                })
            else:
                # Fallback for any unlisted type
                result.append({
                    'injury': injury_type,
                    'severity': data['severity'],
                    'phase': 'General',
                    'icon': '⚠️',
                    'reasons': data['reasons'],
                    'suggestions': [
                        'Consult a qualified sports physiotherapist for a detailed assessment',
                        'Reduce bowling load and monitor for pain or discomfort',
                        'Focus on general strength and conditioning for the affected area',
                    ],
                    'occurrences': data['count']
                })

        return result


class BowlingBiomechanicsAnalyzer:
    def __init__(self, config=None):
        self.config = config or BowlingAnalysisConfig()
        self.injury_analyzer = InjuryRiskAnalyzer(self.config)
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        self._init_state()

    def _init_state(self):
        self.prev = {
            "l_ankle_y": None, "r_ankle_y": None,
            "hip_x": None, "wrist_x": None, "wrist_y": None,
            "l_ankle_x": None, "r_ankle_x": None
        }
        self.buffers = {
            'l_vel': deque(maxlen=8), 'r_vel': deque(maxlen=8),
            'hip_vel': deque(maxlen=8), 'wrist_speed': deque(maxlen=8),
            'elbow_ext': deque(maxlen=30), 'knee_angle': deque(maxlen=10),
            'back_angle': deque(maxlen=10)
        }
        self.thresholds = {"jump_v": None, "plant_v": None, "runup_h": None, "wrist_peak_factor": 3.0}
        self.phase = "INIT"
        self.delivery_counter = 0
        self.deliveries = []
        self.current_delivery = None
        self.frame_idx = 0
        self.calib_samples = {"l_ankle_v": [], "r_ankle_v": [], "hip_h_v": [], "wrist_speed": []}
        self.debug_data = {'time': [], 'lv': [], 'rv': [], 'hipv': [],
                           'wrist': [], 'elbow': [], 'phase': [], 'knee': [], 'back': []}

    def analyze_video(self, video_path, output_dir, progress_callback=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        output_video = os.path.join(output_dir, "annotated_bowling.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

        calib_frames = int(max(1, round(self.config.calibration_seconds * fps)))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(int((self.frame_idx / total_frames) * 85))
            frame = self._process_frame(frame, fps, calib_frames, frame_w, frame_h)
            out.write(frame)

        cap.release()
        out.release()
        self.pose.close()

        injury_report = self.injury_analyzer.generate_injury_report()
        self._finalize_deliveries()

        graphs = self._generate_plots(output_dir, injury_report)
        self._generate_reports(output_dir, injury_report)

        # Generate specific suggestions based on detected injuries
        suggestions = self.injury_analyzer.generate_suggestions(
            injury_report.get('all_risks', [])
        )

        if progress_callback:
            progress_callback(100)

        return {
            'video_path': output_video,
            'video_url': f"annotated_bowling.mp4",
            'deliveries': self.deliveries,
            'injury_report': injury_report,
            'suggestions': suggestions,
            'graphs': graphs
        }

    def _finalize_deliveries(self):
        """Add ICC legality to each delivery"""
        for d in self.deliveries:
            max_elbow = d.get('max_elbow', 0)
            if max_elbow > 15.0:
                d['legality'] = 'ILLEGAL'
                d['legality_note'] = f'Elbow extension {max_elbow:.1f}° exceeds ICC 15° limit'
            else:
                d['legality'] = 'LEGAL'
                d['legality_note'] = f'Elbow extension {max_elbow:.1f}° within ICC 15° limit'

    def _start_delivery(self, frame, time):
        self.delivery_counter += 1
        self.current_delivery = {
            'delivery_number': self.delivery_counter,
            'start_frame': frame,
            'start_time': time,
            'max_elbow': 0,
            'phase_injury_scores': {},
            'action_type': 'UNKNOWN',
            'bowling_type': 'UNKNOWN'
        }
        self.deliveries.append(self.current_delivery)

    def _process_frame(self, frame, fps, calib_frames, frame_w, frame_h):
        t = self.frame_idx / fps
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(image_rgb)
        self.debug_data['phase'].append(self.phase)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            landmarks = self._extract_landmarks(lm)
            metrics = self._calculate_metrics(landmarks, frame_w, frame_h)
            self._update_buffers(metrics)

            if self.frame_idx <= calib_frames:
                self._collect_calibration_data(metrics)
                if self.frame_idx == calib_frames:
                    self._finalize_calibration()
            
            if self.phase != "INIT":
                self._update_phase(metrics, fps)
            
            self._record_debug_data(t, metrics)
            frame = self._draw_annotations(frame, landmarks, metrics, lm, res)
        else:
            cv2.putText(frame, "NO POSE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame

    def _extract_landmarks(self, lm):
        mp_pose = self.mp_pose.PoseLandmark
        return {
            'l_ankle': lm[mp_pose.LEFT_ANKLE], 'r_ankle': lm[mp_pose.RIGHT_ANKLE],
            'l_hip': lm[mp_pose.LEFT_HIP], 'r_hip': lm[mp_pose.RIGHT_HIP],
            'l_shoulder': lm[mp_pose.LEFT_SHOULDER], 'r_shoulder': lm[mp_pose.RIGHT_SHOULDER],
            'r_elbow': lm[mp_pose.RIGHT_ELBOW], 'r_wrist': lm[mp_pose.RIGHT_WRIST],
            'l_knee': lm[mp_pose.LEFT_KNEE], 'r_knee': lm[mp_pose.RIGHT_KNEE],
            'l_elbow': lm[mp_pose.LEFT_ELBOW], 'nose': lm[mp_pose.NOSE]
        }

    def _calculate_metrics(self, lm, frame_w, frame_h):
        metrics = {}
        if self.prev["l_ankle_y"] is None:
            metrics['v_l'] = 0.0; metrics['v_r'] = 0.0
            metrics['hip_h'] = 0.0; metrics['ankle_symmetry'] = 0.0
        else:
            metrics['v_l'] = self.prev["l_ankle_y"] - lm['l_ankle'].y
            metrics['v_r'] = self.prev["r_ankle_y"] - lm['r_ankle'].y
            hip_x = (lm['r_hip'].x + lm['l_hip'].x) / 2
            metrics['hip_h'] = abs(hip_x - self.prev["hip_x"])
            if self.prev["l_ankle_x"] is not None:
                l_move = abs(lm['l_ankle'].x - self.prev["l_ankle_x"])
                r_move = abs(lm['r_ankle'].x - self.prev["r_ankle_x"])
                metrics['ankle_symmetry'] = abs(l_move - r_move) / max(l_move + r_move, 1e-6)
            else:
                metrics['ankle_symmetry'] = 0.0
        self.prev["l_ankle_y"] = lm['l_ankle'].y
        self.prev["r_ankle_y"] = lm['r_ankle'].y
        self.prev["hip_x"] = (lm['r_hip'].x + lm['l_hip'].x) / 2
        self.prev["l_ankle_x"] = lm['l_ankle'].x
        self.prev["r_ankle_x"] = lm['r_ankle'].x
        if self.prev["wrist_x"] is None:
            metrics['wspeed'] = 0.0
        else:
            dx = lm['r_wrist'].x - self.prev["wrist_x"]
            dy = lm['r_wrist'].y - self.prev["wrist_y"]
            metrics['wspeed'] = math.hypot(dx, dy)
        self.prev["wrist_x"] = lm['r_wrist'].x
        self.prev["wrist_y"] = lm['r_wrist'].y
        metrics['elbow_angle'] = self._angle(lm['r_shoulder'], lm['r_elbow'], lm['r_wrist'])
        metrics['elbow_ext'] = abs(180.0 - metrics['elbow_angle'])
        metrics['knee_angle'] = self._angle(lm['r_hip'], lm['r_knee'], lm['r_ankle'])
        metrics['back_angle'] = self._angle(lm['r_shoulder'], lm['r_hip'], lm['r_knee'])
        metrics['shoulder_disp'] = abs(lm['r_shoulder'].x - lm['l_shoulder'].x)
        hip_angle = self._angle(lm['l_hip'], lm['r_hip'], lm['r_shoulder'])
        metrics['hip_rotation'] = abs(90 - hip_angle)
        metrics['jump_height'] = max(metrics['v_l'], metrics['v_r'])
        metrics['landing_velocity'] = abs(metrics['v_l']) + abs(metrics['v_r'])
        return metrics

    def _angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        ba = a - b; bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return math.degrees(math.acos(cosv))

    def _update_buffers(self, metrics):
        self.buffers['l_vel'].append(metrics['v_l'])
        self.buffers['r_vel'].append(metrics['v_r'])
        self.buffers['hip_vel'].append(metrics['hip_h'])
        self.buffers['wrist_speed'].append(metrics['wspeed'])
        self.buffers['elbow_ext'].append(metrics['elbow_ext'])
        self.buffers['knee_angle'].append(metrics['knee_angle'])
        self.buffers['back_angle'].append(metrics['back_angle'])

    def _collect_calibration_data(self, metrics):
        self.calib_samples["l_ankle_v"].append(metrics['v_l'])
        self.calib_samples["r_ankle_v"].append(metrics['v_r'])
        self.calib_samples["hip_h_v"].append(metrics['hip_h'])
        self.calib_samples["wrist_speed"].append(metrics['wspeed'])

    def _finalize_calibration(self):
        med_l = statistics.median(self.calib_samples["l_ankle_v"]) if self.calib_samples["l_ankle_v"] else 0.0
        med_r = statistics.median(self.calib_samples["r_ankle_v"]) if self.calib_samples["r_ankle_v"] else 0.0
        med_hip = statistics.median(self.calib_samples["hip_h_v"]) if self.calib_samples["hip_h_v"] else 0.0
        mad_l = statistics.median([abs(x - med_l) for x in self.calib_samples["l_ankle_v"]]) if self.calib_samples["l_ankle_v"] else 0.0
        mad_r = statistics.median([abs(x - med_r) for x in self.calib_samples["r_ankle_v"]]) if self.calib_samples["r_ankle_v"] else 0.0
        mad_hip = statistics.median([abs(x - med_hip) for x in self.calib_samples["hip_h_v"]]) if self.calib_samples["hip_h_v"] else 0.0
        self.thresholds["jump_v"] = max(0.008, med_l + med_r + 2.0 * max(mad_l, mad_r))
        self.thresholds["plant_v"] = max(0.003, 1.2 * max(mad_l, mad_r))
        self.thresholds["runup_h"] = max(0.006, med_hip + 2.5 * mad_hip)
        if med_hip > self.thresholds["runup_h"] * 0.8:
            self.phase = "RUN-UP"
        else:
            self.phase = "GATHER"

    def _update_phase(self, metrics, fps):
        avg_l = float(np.mean(self.buffers['l_vel'])) if self.buffers['l_vel'] else 0.0
        avg_r = float(np.mean(self.buffers['r_vel'])) if self.buffers['r_vel'] else 0.0
        avg_hip = float(np.mean(self.buffers['hip_vel'])) if self.buffers['hip_vel'] else 0.0
        avg_wrist = float(np.mean(self.buffers['wrist_speed'])) if self.buffers['wrist_speed'] else 0.0
        jth = self.thresholds["jump_v"]
        pth = self.thresholds["plant_v"]
        runth = self.thresholds["runup_h"]
        wrist_baseline = statistics.median(self.calib_samples["wrist_speed"]) if self.calib_samples["wrist_speed"] else 0.0
        wrist_std = np.std(self.calib_samples["wrist_speed"]) if self.calib_samples["wrist_speed"] else 0.0
        wrist_peak_threshold = max(1e-6, wrist_baseline * self.thresholds["wrist_peak_factor"] + 0.5 * wrist_std)

        if self.phase == "RUN-UP":
            if avg_hip < runth * 0.7:
                if self.current_delivery:
                    stride_var = np.std(list(self.buffers['hip_vel'])) if self.buffers['hip_vel'] else 0.0
                    score, risks = self.injury_analyzer.assess_runup_phase(avg_hip, metrics['ankle_symmetry'], stride_var)
                    self.injury_analyzer.phase_risks['RUN-UP'].append({'score': score, 'factors': risks, 'frame': self.frame_idx})
                self.phase = "GATHER"
        elif self.phase == "GATHER":
            if avg_l > jth and avg_r > jth:
                self.phase = "JUMP"
                self._start_delivery(self.frame_idx, self.frame_idx / fps)
        elif self.phase == "JUMP":
            if abs(avg_l) < pth and abs(avg_r) < pth:
                if self.current_delivery:
                    score, risks = self.injury_analyzer.assess_jump_phase(metrics['jump_height'], max(avg_l, avg_r), {'inversion': 0})
                    self.injury_analyzer.phase_risks['JUMP'].append({'score': score, 'factors': risks, 'frame': self.frame_idx})
                self.phase = "PLANT"
                if self.current_delivery:
                    self.current_delivery["plant_frame"] = self.frame_idx
        elif self.phase == "PLANT":
            if self.current_delivery:
                hip_drop = abs(metrics['v_l'] - metrics['v_r'])
                score, risks = self.injury_analyzer.assess_landing_phase(metrics['landing_velocity'], metrics['knee_angle'], hip_drop)
                self.injury_analyzer.phase_risks['PLANT'].append({'score': score, 'factors': risks, 'frame': self.frame_idx})
                self.current_delivery["action_type"] = "FRONT-ON" if metrics['shoulder_disp'] < 0.06 else "SIDE-ON"
            self.phase = "DELIVERY_STRIDE"
            self.delivery_counter = self.config.delivery_window_frames
        elif self.phase == "DELIVERY_STRIDE":
            self.delivery_counter = max(0, self.delivery_counter - 1)
            if metrics['wspeed'] > wrist_peak_threshold or (metrics['wspeed'] > 2.0 * avg_wrist):
                self.phase = "BALL_RELEASE"
                if self.current_delivery:
                    self.current_delivery["release_frame"] = self.frame_idx
                    score, risks = self.injury_analyzer.assess_delivery_phase(
                        metrics['elbow_ext'], metrics['back_angle'], 160.0,
                        self.current_delivery.get("action_type", "UNKNOWN"), metrics['hip_rotation'])
                    self.injury_analyzer.phase_risks['DELIVERY_STRIDE'].append({'score': score, 'factors': risks, 'frame': self.frame_idx})
        elif self.phase == "BALL_RELEASE":
            if self.current_delivery:
                if metrics['elbow_ext'] > self.current_delivery.get("max_elbow", 0):
                    self.current_delivery["max_elbow"] = metrics['elbow_ext']
            if self.delivery_counter <= 0:
                self.phase = "FOLLOW_THROUGH"
        elif self.phase == "FOLLOW_THROUGH":
            if self.current_delivery:
                decel = abs(metrics['wspeed'] - (float(np.mean(self.buffers['wrist_speed'])) if self.buffers['wrist_speed'] else 0.0))
                trunk_rot = abs(90 - metrics['back_angle'])
                score, risks = self.injury_analyzer.assess_followthrough_phase(decel, metrics['wspeed'], trunk_rot)
                self.injury_analyzer.phase_risks['FOLLOW_THROUGH'].append({'score': score, 'factors': risks, 'frame': self.frame_idx})
                # Bowling type detection
                if self.current_delivery.get("max_elbow", 0) < 5 and metrics['wspeed'] < 0.05:
                    self.current_delivery["bowling_type"] = "SPIN"
                elif self.current_delivery.get("max_elbow", 0) > 10:
                    self.current_delivery["bowling_type"] = "FAST"
                else:
                    self.current_delivery["bowling_type"] = "MEDIUM"
                # Phase injury scores
                phase_scores = {}
                for ph, risks_list in self.injury_analyzer.phase_risks.items():
                    if risks_list:
                        phase_scores[ph] = round(sum(r['score'] for r in risks_list) / len(risks_list), 1)
                self.current_delivery["phase_injury_scores"] = phase_scores
                overall = round(sum(phase_scores.values()) / max(len(phase_scores), 1), 1) if phase_scores else 0
                self.current_delivery["overall_injury_score"] = overall
            self.phase = "GATHER"
            self.current_delivery = None

    def _record_debug_data(self, t, metrics):
        self.debug_data['time'].append(t)
        self.debug_data['lv'].append(metrics['v_l'])
        self.debug_data['rv'].append(metrics['v_r'])
        self.debug_data['hipv'].append(metrics['hip_h'])
        self.debug_data['wrist'].append(metrics['wspeed'])
        self.debug_data['elbow'].append(metrics['elbow_ext'])
        self.debug_data['knee'].append(metrics['knee_angle'])
        self.debug_data['back'].append(metrics['back_angle'])

    def _draw_annotations(self, frame, landmarks, metrics, lm, res):
        self.mp_draw.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2))
        # Phase and metrics overlay
        phase_color = {'RUN-UP': (0,255,0), 'GATHER': (255,255,0), 'JUMP': (0,165,255),
                       'PLANT': (0,0,255), 'DELIVERY_STRIDE': (255,0,255),
                       'BALL_RELEASE': (0,100,255), 'FOLLOW_THROUGH': (255,200,0)}.get(self.phase, (200,200,200))
        cv2.putText(frame, f'Phase: {self.phase}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, phase_color, 2)
        cv2.putText(frame, f'Elbow Ext: {metrics["elbow_ext"]:.1f}°', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
        cv2.putText(frame, f'Knee: {metrics["knee_angle"]:.1f}°', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        # ICC legality indicator
        elbow_ext = metrics['elbow_ext']
        if elbow_ext > 15:
            cv2.putText(frame, f'ICC: ILLEGAL ({elbow_ext:.1f}°)', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'ICC: LEGAL ({elbow_ext:.1f}°)', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Deliveries: {len(self.deliveries)}', (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return frame

    def _generate_plots(self, output_dir, injury_report):
        graphs = []
        try:
            # 1. Elbow extension over time
            if self.debug_data['time']:
                plt.figure(figsize=(12, 4))
                plt.plot(self.debug_data['time'], self.debug_data['elbow'], color='cyan', linewidth=2, label='Elbow Extension')
                plt.axhline(15, color='red', linestyle='--', label='ICC Legal Limit (15°)', linewidth=2)
                plt.xlabel("Time (s)"); plt.ylabel("Elbow Extension (°)")
                plt.title("Elbow Extension Over Time (ICC Legality Check)")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                p = os.path.join(output_dir, "elbow_extension.png")
                plt.savefig(p, dpi=150, facecolor='#0a0a0f'); plt.close()
                graphs.append({'name': 'Elbow Extension (ICC Check)', 'url': 'elbow_extension.png'})

            # 2. Joint angles
            if self.debug_data['time']:
                plt.figure(figsize=(12, 5))
                plt.plot(self.debug_data['time'], self.debug_data['elbow'], label="Elbow Ext", color='cyan', alpha=0.8)
                plt.plot(self.debug_data['time'], self.debug_data['knee'], label="Knee Angle", color='magenta', alpha=0.7)
                plt.plot(self.debug_data['time'], self.debug_data['back'], label="Back Angle", color='lime', alpha=0.7)
                plt.axhline(15, color='red', linestyle=':', alpha=0.5, label='Elbow Limit (15°)')
                plt.xlabel("Time (s)"); plt.ylabel("Angle (degrees)")
                plt.title("Joint Angle Analysis"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                p = os.path.join(output_dir, "joint_angles.png")
                plt.savefig(p, dpi=150); plt.close()
                graphs.append({'name': 'Joint Angles', 'url': 'joint_angles.png'})

            # 3. Injury risk by phase
            if injury_report.get('phase_scores'):
                plt.figure(figsize=(10, 5))
                phases = list(injury_report['phase_scores'].keys())
                scores = list(injury_report['phase_scores'].values())
                colors = ['#00ff41' if s < 30 else '#ffff00' if s < 60 else '#ff0040' for s in scores]
                plt.barh(phases, scores, color=colors, alpha=0.85)
                plt.xlabel("Injury Risk Score"); plt.title("Injury Risk by Phase")
                plt.xlim(0, 100)
                for i, (ph, sc) in enumerate(zip(phases, scores)):
                    plt.text(sc + 2, i, f'{sc:.1f}', va='center', color='white')
                plt.tight_layout()
                p = os.path.join(output_dir, "injury_risk_by_phase.png")
                plt.savefig(p, dpi=150); plt.close()
                graphs.append({'name': 'Injury Risk by Phase', 'url': 'injury_risk_by_phase.png'})

            # 4. Delivery comparison
            if len(self.deliveries) > 0:
                delivery_nums = [d['delivery_number'] for d in self.deliveries]
                elbows = [d.get('max_elbow', 0) for d in self.deliveries]
                injury_scores = [d.get('overall_injury_score', 0) for d in self.deliveries]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                ax1.bar(delivery_nums, elbows, color=['#ff0040' if e > 15 else '#00ff41' for e in elbows], alpha=0.8)
                ax1.axhline(15, color='red', linestyle='--', label='Legal Limit (15°)')
                ax1.set_ylabel("Max Elbow Extension (°)"); ax1.set_title("Elbow Extension by Delivery")
                ax1.legend(); ax1.grid(alpha=0.3)
                ax2.bar(delivery_nums, injury_scores,
                        color=['#00ff41' if s < 30 else '#ffff00' if s < 60 else '#ff0040' for s in injury_scores], alpha=0.8)
                ax2.set_xlabel("Delivery Number"); ax2.set_ylabel("Injury Risk Score")
                ax2.set_title("Overall Injury Risk by Delivery"); ax2.grid(alpha=0.3)
                plt.tight_layout()
                p = os.path.join(output_dir, "delivery_comparison.png")
                plt.savefig(p, dpi=150); plt.close()
                graphs.append({'name': 'Delivery Comparison', 'url': 'delivery_comparison.png'})

        except Exception as e:
            print(f"Plot error: {e}")
        return graphs

    def _generate_reports(self, output_dir, injury_report):
        try:
            # CSV
            csv_path = os.path.join(output_dir, "delivery_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Delivery', 'Action Type', 'Bowling Type', 'Max Elbow (°)', 'ICC Legality', 'Overall Risk'])
                for d in self.deliveries:
                    writer.writerow([d['delivery_number'], d.get('action_type', 'N/A'),
                                     d.get('bowling_type', 'N/A'), f"{d.get('max_elbow', 0):.1f}",
                                     d.get('legality', 'N/A'), f"{d.get('overall_injury_score', 0):.1f}"])
            # JSON
            json_path = os.path.join(output_dir, "detailed_analysis.json")
            with open(json_path, 'w') as f:
                json.dump({'deliveries': self.deliveries, 'injury_report': {
                    'overall_score': injury_report.get('overall_score'),
                    'risk_level': injury_report.get('risk_level'),
                    'phase_scores': injury_report.get('phase_scores')
                }}, f, indent=2)
        except Exception as e:
            print(f"Report error: {e}")


class BowlingAnalysisModule:
    def analyze(self, video_path, output_dir, progress_callback=None):
        config = BowlingAnalysisConfig(output_dir=output_dir)
        analyzer = BowlingBiomechanicsAnalyzer(config)
        return analyzer.analyze_video(video_path, output_dir, progress_callback)

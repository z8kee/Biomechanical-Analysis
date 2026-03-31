#Any rules regarding sprinting form will be placed and fetched from here.
import numpy as np, requests

LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

THRESHOLDS = {
    "backside_time": 0.12,
    "backside_distance": 0.10,
    "heel_recovery": 0.15
}

def get_point(frame, landmark_id):
    index = landmark_id*4
    return np.array(frame[index:index+3])

def distance(a,b):
    return np.linalg.norm(a-b)

def angle_abc(a,b,c):
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))

def velocity(p_prev, p_curr, dt):
    return (p_curr - p_prev) / dt

def angular_velocity(angle_curr, angle_prev, dt):
    return (angle_curr - angle_prev) / dt

def torso_angle(frame, side):
    hip = get_point(frame, LANDMARKS[f"{side}_hip"])
    shoulder = get_point(frame, LANDMARKS[f"{side}_shoulder"])
    torso_vector = shoulder - hip
    vert = np.array([0, -1, 0])

    cos_angle = np.dot(torso_vector, vert) / (
        np.linalg.norm(torso_vector) * np.linalg.norm(vert) + 1e-6
    )
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return angle > 50, angle

def excessive_backside(frames, fps, HIP, ANKLE, LEG_LENGTH):
    linger = 0

    for f in frames:
        hip = get_point(f, HIP)
        foot = get_point(f, ANKLE)

        if foot[0] < hip[0]:
            dist = (hip[0] - foot[0]) / distance(HIP, ANKLE)
            if dist > THRESHOLDS["backside_distance"]:
                linger += 1
        else:
            linger = 0

    return (linger / fps) > THRESHOLDS["backside_time"]

def shin_chest_angle(frame, side):
    knee = get_point(frame, LANDMARKS[f"{side}_knee"])
    ankle = get_point(frame, LANDMARKS[f"{side}_ankle"])
    shin = knee - ankle
    vert = np.array([0, -1, 0])

    cos_angle = np.dot(shin, vert) / (
        np.linalg.norm(shin) * np.linalg.norm(vert) + 1e-6
    )

    deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    angtorso, torso = torso_angle(frame, side)

    return 0.95 * torso <= deg <= 1.05 * torso, deg

def dorsiflexion_angle(frame, side):
    f_in = get_point(frame, LANDMARKS[f"{side}_foot_index"])
    heel = get_point(frame, LANDMARKS[f"{side}_heel"])

    dorsi = f_in - heel
    horz = np.array([-1, 0, 0])

    cos_angle = np.dot(dorsi, horz) / (
        np.linalg.norm(dorsi) * np.linalg.norm(horz) + 1e-6
    )
    
    return float(np.degrees(np.clip(cos_angle, -1.0, -1.0)))


def analyse_form(predictions, inputs, fps):
    flags = []

    data = inputs.cpu().numpy()

    for i, phase_label in enumerate(predictions):
        curr_frame = data[i][15]
        if i > 0:
            prev_frame = data[i-1][15]
        if phase_label == 0:

            high, torsoang = torso_angle(curr_frame, "left") or torso_angle(curr_frame, "right")
            if high:
                flags.append(f"Start: Torso rising too early loses horizontal force.")

            rknee = get_point(curr_frame, LANDMARKS["right_knee"])
            rheel = get_point(curr_frame, LANDMARKS["right_heel"])
            lknee = get_point(curr_frame, LANDMARKS["left_knee"])
            lheel = get_point(curr_frame, LANDMARKS["left_heel"])
            if rknee[1] < rheel[1]: flags.append("Right heel recovery too high causes wasted time.")
            if lknee[1] < lheel[1]: flags.append("Left heel recovery too high causes wasted time.")

        elif phase_label == 1:
            is_parallel, angle_val = shin_chest_angle(curr_frame, "left") or shin_chest_angle(curr_frame, "right")
            if not is_parallel:
                flags.append(f"Acceleration: Poor shin-to-chest alignment detected.")

            elif angular_velocity(torso_angle(curr_frame, "left")[1], torso_angle(prev_frame, "left")[1], 1/fps) > 10:
                flags.append("Acceleration: Standing up too fast.")
            
            elif angle_abc(get_point(curr_frame, LANDMARKS["left_shoulder"]),
                           get_point(curr_frame, LANDMARKS["left_hip"]),
                           get_point(curr_frame, LANDMARKS["left_knee"])) < 10 or angle_abc(get_point(curr_frame, LANDMARKS["right_shoulder"]),
                                                           get_point(curr_frame, LANDMARKS["right_hip"]),
                                                           get_point(curr_frame, LANDMARKS["right_knee"])) < 10:
                flags.append("Acceleration: Excessive hip flexion detected. Knee lift too high, too early.")
            
            elif excessive_backside(data[i], fps, LANDMARKS["left_hip"], LANDMARKS["left_ankle"],
                                     distance(get_point(curr_frame, LANDMARKS["left_hip"]), 
                                              get_point(curr_frame, LANDMARKS["left_ankle"]))) or excessive_backside(data[i], fps, LANDMARKS["right_hip"], LANDMARKS["right_ankle"],
                                                                                                                      distance(get_point(curr_frame, LANDMARKS["right_hip"]), 
                                                                                                                               get_point(curr_frame, LANDMARKS["right_ankle"]))):
                flags.append("Backside: Excessive backside mechanics detected.")

        elif phase_label == 2:
            rfoot = get_point(curr_frame, LANDMARKS["right_foot_index"])
            lfoot = get_point(curr_frame, LANDMARKS["left_foot_index"])
            lhip = get_point(curr_frame, LANDMARKS["left_hip"])
            rhip = get_point(curr_frame, LANDMARKS["right_hip"])
            if distance(rfoot[0], lhip[0]) > 0.15 or distance(lfoot[0], rhip[0]) > 0.15:
                flags.append("Overstriding detected, reduces efficiency and increases injury risk.")

            elif dorsiflexion_angle(curr_frame, "left") or dorsiflexion_angle(curr_frame, "right") > 15.0:
                flags.append("Poor ankle stiffness")

    return list(set(flags))

def more_detailed_feedback(flags, key):
    base_url = 'https://api.openai.com/v1/chat/completions'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    payload = {
        "model": "gpt-5-nano",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert biomechanics coach specializing in sprinting form analysis. Provide detailed feedback and corrective exercises based on biomechanical issues. Try not to use buzzwords and be more simplistic so the average sprinter/runner or even a pro athlete can understand clearly."
            },
            {
                "role": "user",
                "content": f"The following biomechanical issues were detected during a sprint analysis: {flags}. For each issue, provide an explanation of why it is problematic and suggest specific drills or exercises to correct it. Make it a slightly short."
            }
        ]
    }

    response = requests.post(base_url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed due to {response.status_code}: {response.text}")
    
    result = response.json()

    return result['choices'][0]['message']['content']

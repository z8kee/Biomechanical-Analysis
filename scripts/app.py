import streamlit as st, tempfile, torch, os, pandas as pd, time
from featextract import extract_features_for_app
from phaseclassifier import PhaseClassifier
from biomechanics import analyse_form, more_detailed_feedback


st.set_page_config("Biomechanical Analysis", layout="centered")

if 'data' not in st.session_state:
    st.session_state['data'] = None

model_path = 'models/phase_classifier.pth'
PHASE_TO_LABEL = {0: "Start",
             1: "Acceleration",
             2: "Max Velocity",
             3: "Deceleration",
             4: "Transition"}

st.title("PaceyAI")
upload = st.file_uploader("Upload running video", type=['mp4', 'mov'])

if upload is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    tfile.close()

    st.video(tfile.name)

    if st.button("Analyse Sprint"):

        with st.spinner("Extracting features"):
            input_ten, fps, _ = extract_features_for_app(tfile.name)
            
            if input_ten is not None:

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = PhaseClassifier().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                outputs = model(input_ten.to(device))
                preds = torch.argmax(outputs, dim=1)
                flags = analyse_form(preds, input_ten, fps)
                

                st.session_state['data'] = {
                    'preds': preds,
                    'fps': fps,
                    'flags': flags,
                    'gpt_feedback': None # placeholder
                }
                st.success("Analysis Complete")
            else:
                st.error("Could not extract data.")
        
        # cleanup file
        os.unlink(tfile.name)


    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        # Display Timeline
        st.subheader("Phase Timeline")
        timeline = []
        for i, p in enumerate(data['preds']):
            t_start, t_end = (i*15)/data['fps'], (i*15+30)/data['fps']
            timeline.append({"Time": f"{t_start:.1f}-{t_end:.1f}s",
                            "Phase": PHASE_TO_LABEL[int(p.item())]})
        st.dataframe(pd.DataFrame(timeline), height=200)


        st.subheader("Form Imperfections")
        if not data['flags']:
            st.write("Form looks good!")

        else:

            for f in data['flags']: 
                st.write(f"- {f}")

            
            if st.button("More Feedback"):
                with st.spinner("Asking Coach..."):
                    data['gpt_feedback'] = more_detailed_feedback(data['flags'], os.getenv("SPRINTKEY"))
                    
            # Show Feedback if it exists
            if data['gpt_feedback']:
                st.info(data['gpt_feedback'])

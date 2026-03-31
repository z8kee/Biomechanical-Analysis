import streamlit as st, tempfile, torch, os, pandas as pd
from scripts.frameextract import extract_features_for_app
from scripts.phaseclassifier import PhaseClassifier
from scripts.biomechanics import analyse_form, more_detailed_feedback


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
upload = st.file_uploader("Upload running video, preferably side view video.", type=['mp4', 'mov'])

if upload is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    tfile.close()

    st.video(tfile.name)

    if st.button("Analyse Sprint"):

        with st.spinner("Extracting features"):
            # Check if model file exists
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
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
                        'gpt_feedback': None
                    }
                    st.success("Analysis Complete")
                else:
                    st.error("Could not extract data. Check logs for details.")
        
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
                    data['gpt_feedback'] = more_detailed_feedback(data['flags'], st.secrets["OPENAI_API_KEY"])
                    
            # Show Feedback if it exists
            if data['gpt_feedback']:
                st.info(data['gpt_feedback'])


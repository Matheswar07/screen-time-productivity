import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Screen-Time Productivity Profiler",
    page_icon="📱",
    layout="centered"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 1.5rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #9aa0a6;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Input card */
    .input-card {
        padding: 1rem 1.2rem 0.7rem 1.2rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.02);
    }

    .input-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    .input-description {
        color: #8f969d;
        font-size: 0.78rem;
        margin-bottom: 0.55rem;
    }

    /* Prediction result */
    .result-box {
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.2rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .result-title {
        font-size: 0.9rem;
        color: #9aa0a6;
        margin-bottom: 0.3rem;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Section title */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.8rem;
        margin-bottom: 0.9rem;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        height: 3.1rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD SAVED ML COMPONENTS
# ==========================================================

kmeans = joblib.load(
    "model/kmeans_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

cluster_profiles = joblib.load(
    "model/cluster_profiles.pkl"
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">📱 Screen-Time Productivity Profiler</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze daily digital habits and identify a productivity profile'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown(
    '<div class="section-title">📊 Daily Behaviour</div>',
    unsafe_allow_html=True
)


# ==========================================================
# ROW 1
# ==========================================================

col1, col2 = st.columns(2)


with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="input-title">📱 Screen Time</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-description">'
            'Total daily screen usage'
            '</div>',
            unsafe_allow_html=True
        )

        screen_time = st.number_input(
            "Screen Time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            label_visibility="collapsed"
        )


with col2:

    with st.container(border=True):

        st.markdown(
            '<div class="input-title">📲 Social Media</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-description">'
            'Time spent on social media'
            '</div>',
            unsafe_allow_html=True
        )

        social_media = st.number_input(
            "Social Media Usage (hours)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            label_visibility="collapsed"
        )


# ==========================================================
# ROW 2
# ==========================================================

col1, col2 = st.columns(2)


with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="input-title">🎮 Entertainment</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-description">'
            'Gaming, videos and other entertainment'
            '</div>',
            unsafe_allow_html=True
        )

        entertainment = st.number_input(
            "Entertainment Usage (hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.5,
            step=0.5,
            label_visibility="collapsed"
        )


with col2:

    with st.container(border=True):

        st.markdown(
            '<div class="input-title">📚 Productive Phone Use</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-description">'
            'PDFs, classes, research, coding and study'
            '</div>',
            unsafe_allow_html=True
        )

        productive_phone_use = st.number_input(
            "Productive Phone Use (hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
            label_visibility="collapsed"
        )


# ==========================================================
# ROW 3
# ==========================================================

col1, col2 = st.columns(2)


with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="input-title">📖 Study Time</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-description">'
            'Total time spent studying'
            '</div>',
            unsafe_allow_html=True
        )

        study_time = st.number_input(
            "Study Time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            label_visibility="collapsed"
        )


with col2:

    st.empty()


# ==========================================================
# VALIDATION
# ==========================================================

total_activity_time = (
    social_media
    + productive_phone_use
    + entertainment
)

activity_exceeds_screen_time = total_activity_time > screen_time
productive_use_exceeds_study_time = productive_phone_use > study_time

if activity_exceeds_screen_time:

    st.error(
        "⚠️ Social Media, Productive Phone Use, and Entertainment hours combined cannot exceed Screen Time hours."
    )

if productive_use_exceeds_study_time:

    st.error(
        "⚠️ Productive Phone Use cannot be greater than Study Time."
    )


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

st.write("")

predict = st.button(
    "🔍 Predict Productivity Profile",
    type="primary"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    if activity_exceeds_screen_time:

        st.error(
            "⚠️ Social Media, Productive Phone Use, and Entertainment hours combined cannot exceed Screen Time hours."
        )

    elif productive_use_exceeds_study_time:

        st.error(
            "Please make sure Productive Phone Use is "
            "less than or equal to Study Time."
        )

    else:

        # --------------------------------------------------
        # Create input DataFrame
        # --------------------------------------------------

        input_data = pd.DataFrame(
            [[
                screen_time,
                social_media,
                entertainment,
                productive_phone_use,
                study_time
            ]],
            columns=[
                "Screen_Time_hr",
                "Social_Media_hr",
                "Entertainment_hr",
                "Productive_Phone_Use_hr",
                "Study_Time_hr"
            ]
        )


        # --------------------------------------------------
        # Scale input
        # --------------------------------------------------

        scaled_input = scaler.transform(
            input_data
        )


        # --------------------------------------------------
        # Predict cluster
        # --------------------------------------------------

        predicted_cluster = int(
            kmeans.predict(
                scaled_input
            )[0]
        )


        # --------------------------------------------------
        # Get profile
        # --------------------------------------------------

        predicted_profile = cluster_profiles[
            predicted_cluster
        ]


        # ==================================================
        # RESULT
        # ==================================================

        profile_class = predicted_profile.lower()

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">
                    Productivity Profile
                </div>
                <div class="result-value">
                    {predicted_profile.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ==================================================
        # YOUR INPUTS
        # ==================================================

        st.markdown(
            '<div class="section-title">📝 Your Inputs</div>',
            unsafe_allow_html=True
        )


        input_display = pd.DataFrame(
            {
                "Parameter": [
                    "📱 Screen Time",
                    "📲 Social Media",
                    "🎮 Entertainment",
                    "📚 Productive Phone Use",
                    "📖 Study Time"
                ],

                "Value": [
                    f"{screen_time:.1f} hr",
                    f"{social_media:.1f} hr",
                    f"{entertainment:.1f} hr",
                    f"{productive_phone_use:.1f} hr",
                    f"{study_time:.1f} hr"
                ]
            }
        )


        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True
        )

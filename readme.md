# Artsy

## Description
Artsy is a project implemented using Cyberwave's SO-101 robotic arm.
It is designed to automate daily makeup routines, some of which tend to be tasking when done by human beings.

The user issues a voice/text command e.g "apply red lipstick" or "draw some dark-blue eyebrows". If the command is issued as speech it first gets translated to text using Mistral speech-to-text engine. Deepseek LLM then translates the text command into a structured joint-motion plan, and the Cyberwave SDK executes it on the digital twin. 

## Tech stack

- **STT** — Mistral Voxtral (`voxtral-mini-latest`), spacebar push-to-talk.
- **Planner** — Deepseek (text-only) → constrained JSON action plan.
- **Robot control** — Cyberwave Python SDK → MQTT → SO-101 follower.
- **MediaPipe** — Detects facial features


## Quick start

1. Clone the project
    ```bash
    git clone https://github.com/Rabinnnn/artsy.git
    ```
- It is recommended that you create and activate a python virtual environment before proceeding to the next step.

2. Configure the environment
    ```bash
    cd nl_arm_controller

    cp .env.example .env
    # edit .env with your three API keys

    pip install -r requirements.txt

    ```
3. Start the server
    ```bash
    python server.py
    ```

4. Run the digital twin
    ```bash
    # if you intend to issue text commands by typing on the terminal
    python nl_arm_controller.py 
    # OR
    # if you intend to issue commands via speech
    python nl_arm_controller.py --voice
    ```
- At this point you can enter the allowed commands and you will be able to see the digital twin perform the actions in the Cyberwave simulation environment. However, at this point you will only be able to see the robot's motion since there is no face model on the environment that can be use to visualize how the makeup is being applied. To work around this, proceed to the next step below, which uses an external web app to show how the makeup is being applied. 

5. Launch the web app
- You can do this by simply opening the index.html file on your browser or use the "Go Live" option found on Vs code.
You should see a web interface as shown below. 


  ![alt text](<artsy 1.png>)


- Next, upload your image of a face model. Make sure the image is clear enough to make it possible to detect facial features. For a start, consider using the one provide in this repo, face.jpg, since it provides a detailed 3D render of the face.

6. Provide commands
- Proceed to issue commands in the terminal where you ran "python nl_arm_controller.py". You can either type them manually or speak them out (depending on whether you included the --voice flag or not).
- Currently only 3 actions are supported; applying lipstick, drawing eyebrows, and applying blush on the cheeks. Make sure you specify the color that you want e.g apply some purple lipstick.
- Open the Cyberwave simulation environment alongside the browser that is running your web app so that you can see both the motion of the robot and the makeup being applied on the image.

  ![alt text](<Screenshot 2026-08-05 120806-1.png>)
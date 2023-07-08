# Micro-Speech-To-Text (Micro-STT)

Includes setup for strict typechecking with [mypy](https://mypy.readthedocs.io/en/stable/index.html) and linting with [flake8](https://flake8.pycqa.org/en/latest/index.html) + settings for the [VSCode Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

## Raspberry Pi Setup

The project is tested to work on a raspberry Pi 3 Model B with 64-bit Ubuntu Server 20.04.2 LTS installed on an SD card using the [Raspberry Pi Imager](https://www.raspberrypi.com/software).

### PortAudio

To access the Pi's audio devices `PortAudio` must be installed [(src)](https://gist.github.com/tstellanova/11ef60480552e2c5660af8e9e14410c8):

```bash
sudo apt-get install libasound2-dev portaudio19-dev libatlas-base-dev
```

To prevent issues with the audio stream, the microphone input mus be configured [(src)](https://github.com/googlesamples/assistant-sdk-python/issues/382#issuecomment-583847256). In the user's home directory, create an `.asoundrc` file:

```bash
touch ~/.asoundrc
```

And enter the following config:

```txt
pcm.!default {
  type asym
  capture.pcm "mic"
  playback.pcm "speaker"
}
pcm.mic {
  type plug
  slave {
    pcm "hw:1,0"
    rate 48000
  }
}
pcm.speaker {
  type plug
  slave {
    pcm "hw:0,0"
  }
}
```

To check the connected audio devices `alsa-utils` can be installed:

```bash
sudo apt install alsa-utils
```

and the audio devices listed using:

```bash
arecord -l
```

### Access Point

For more convenient access to the Pi a WiFi Access Point can be set up [(src)](https://ubuntu.com/core/docs/networkmanager/configure-wifi-access-points), [(src)](https://netplan.io/reference/#properties-for-device-type-wifis%3A), [(src)](https://linuxconfig.org/ubuntu-20-04-connect-to-wifi-from-command-line).

Install `NetworkManager`:

```bash
sudo apt-get install network-manager
```

And update the `/etc/netplan/50-cloud-init.yaml` (adding your own SSID and password):

```yml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    eth0:
      dhcp4: true
      optional: true
  wifis:
    wlan0:
      dhcp4: true
      optional: true
      # Static IP-address
      addresses: [10.42.0.1/24]
      access-points:
        "<ssid>":
          password: "<pwd>"
          # AP mode
          mode: ap
```

Apply the config using:

```bash
sudo netplan generate
sudo netplan --debug apply
```

And check that your Access Point appears in the `NetworkManager`-Interfaces:

```bash
$ nmcli device

| DEVICE | TYPE     | STATE     | CONNECTION             |
| ------ | -------- | --------- | ---------------------- |
| wlan0  | wifi     | connected | netplan-wlan0-\<ssid\> |
| ...    | ...      | ...       | ...                    |
```

Sometimes a reboot is needed for the Access Point to appear:

```bash
sudo reboot now
```

## Installation (Ubuntu)

__*Uses Python 3.10*__

Create a virtual environment:

```bash
python3 -m venv venv
```

Active the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Create .env file

Create the .env scaffold using

```bash
bash init_dotenv.sh
```

and enter your configuration.

### Run Application

Start CLI application:

```bash
python3 -m app
```

### Run Typechecks and Linter

Run mypy for typechecks:

```bash
mypy -m app
```

Run flake8 for linting:

```bash
flake8
```

## Actions

Overview of the application's actions, callable via the CLI interface

CLI options are automatically prompted for via inquirer, not passed as flags

### Record audio (save as WAV)

- CLI options: `duration`, `filename`
- Record audio using settings defined in `.env`
- Output recording as WAV-file in `./out/recordings`

### Re-record audio files (from WAV)

- CLI options: `input_directory`, `output_directory_name`
- Select folder containing WAV-files form `./in`
- Simultaneously play and record audio files using settings defined in `.env`
- Output recording as WAV-file in subdirectory of `./out/recordings`

### Calibrate noise floor (from WAV)

- CLI options: `recording`
- Select WAV-file from `./out/recordings`
- Set audio as noise floor for entire application

### Visualize preprocessing (from WAV)

- CLI options: `recording`, `save_preprocessed_recording`, `output_directory_name`
- Select WAV-file from `./out/recordings`
- Visualize preprocessing using settings defined in `.env`
- Output visualization in `./out/viz`
- Output preprocessed recording in `./out/recordings` (optional)

### Transcribe recording (from WAV)

- CLI options: `recording`, `model`, `use_preprocessing`
- Select WAV-file from `./out/recordings`
- Print transcription to stdout

### Live transcription (audio stream)

- CLI options: `model`, `use_preprocessing`, `enable_audio_passthrough`
- Start audio stream using settings defined in `.env`
- Print transcription to stdout

### Performance benchmark

- CLI options: `input_directory`, `models`, `micro_controller_config`, `benchmark_name`, `iterations`
- Select folder containing WAV-files form `./in`
- Select JSON micro controller definition from `./in`
- Output results in subdirectory of `./out/benchmarks/performance`

### Quality benchmark

- CLI options: `input_directory`, `target_transcriptions`, `models`, `use_preprocessing`, `benchmark_name`
- Select folder containing WAV-files form `./in`
- Select CSV target transcriptions from `./in`
- Output results in subdirectory of `./out/benchmarks/quality`

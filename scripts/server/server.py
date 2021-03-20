#!/usr/bin/env python3

import click
import os
import flask
import pickle


app = flask.Flask(__name__)
DATA_DIR = None
SAVE_DIR = None

@click.command()
@click.argument("data_dir",
                type=click.Path(exists=True, file_okay=False))
@click.argument("save_dir",
                type=click.Path(file_okay=False))
@click.option('--host', default='0.0.0.0')
@click.option('-p', '--port', type=int, default=8000)
def main(data_dir, save_dir, host, port, users=["BH", "DD"]):
    global DATA_DIR
    global SAVE_DIR
    DATA_DIR = data_dir
    SAVE_DIR = save_dir
    app.run(host=host, port=port, threaded=True, debug=True)

@app.route("/")
def _index():
    return index("")

@app.route("/<string:user>/")
def index(user):
    print(DATA_DIR)
    print(os.listdir(DATA_DIR))
    videos = os.listdir(DATA_DIR)
    videos = list(map(lambda v: os.path.splitext(v)[0], videos))
    return flask.render_template("index.html", user=user, videos=videos)

@app.route("/<string:user>/<string:video>", methods=["GET", "POST"])
def label(user, video):
    output = os.path.join(SAVE_DIR, user, "{}.pkl".format(video))
    if flask.request.method == "GET":
        data = {}
        if os.path.isfile(output):
            try:
                with open(output, "rb") as f:
                    data = pickle.load(f)
            except:
                data = {}

        videos = os.listdir(DATA_DIR)
        videos = list(map(lambda v: os.path.splitext(v)[0], videos))
        index = videos.index(video)
        total = len(videos)
        prev = None
        if index != 0:
            prev = videos[index - 1]
        next = None
        if index + 1 < len(videos):
           next = videos[index + 1]
        return flask.render_template("label.html", user=user, video=video, index=(index + 1), total=total, prev=prev, next=next, data=data)
    else:
        data = flask.request.data
        data = data.strip().decode("utf-8").split("\n")
        print(data)
        data = [d.split(" ") for d in data]
        print(data)
        data = {key: value for (key, value) in data}
        print(data)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        # TODO: save to temp loc and move
        with open(output, "wb") as f:
            pickle.dump(data, f)

        return ""


@app.route("/video.avi")
def video_avi():
    return flask.send_file("/home/bryanhe/dynamic/er_videos_2/VID33910.avi")

@app.route("/video/<string:video>.mp4")
def video_mp4(video):
    return flask.send_file(os.path.abspath(os.path.join(DATA_DIR, "{}.mp4".format(video))))

if __name__ == "__main__":
    main()

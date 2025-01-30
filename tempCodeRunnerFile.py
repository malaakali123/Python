from flask import Flask, render_template, request
import exercises.squats as squats
import exercises.shoulder_press as shoulder_press
import exercises.hand_curls as hand_curls
import exercises.lunges as lunges
import exercises.pushups as pushups
import exercises.situps as situps
import exercises.triceps_curls as triceps_curls
import exercises.lungesr as lungesr
import exercises.delt_raises as delt_raises


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/home.html')
def home1():
    return render_template('home.html')


@app.route('/detect/squats')
def detect_squats():
    result, counter, elapsed_time_formatted, calories_burnt = squats.detect()
    return render_template('results.html', exercise='Squats', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/shoulder_press')
def detect_shoulder_press():
    result, counter, elapsed_time_formatted, calories_burnt = shoulder_press.detect()
    return render_template('results.html', exercise='Shoulder Press', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/hand_curls')
def detect_hand_curls():
    result, counter, elapsed_time_formatted, calories_burnt = hand_curls.detect()
    return render_template('results.html', exercise='Hand Curls', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/lunges')
def detect_lunges():
    result, counter, elapsed_time_formatted, calories_burnt = lunges.detect()
    return render_template('results.html', exercise='lunges', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/pushups')
def detect_pushups():
    result, counter, elapsed_time_formatted, calories_burnt = pushups.detect()
    return render_template('results.html', exercise='pushups', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/situps')
def detect_situps():
    result, counter, elapsed_time_formatted, calories_burnt = situps.detect()
    return render_template('results.html', exercise='situps', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/triceps_curls')
def detect_triceps_curls():
    result, counter, elapsed_time_formatted, calories_burnt = triceps_curls.detect()
    return render_template('results.html', exercise='triceps_curls', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/lungesr')
def detect_lungesr():
    result, counter, elapsed_time_formatted, calories_burnt = lungesr.detect()
    return render_template('results.html', exercise='lungesr', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


@app.route('/detect/delt_raises')
def detect_delt_raises():
    result, counter, elapsed_time_formatted, calories_burnt = delt_raises.detect()
    return render_template('results.html', exercise='delt_raises', result=result, counter=counter, elapsed_time_formatted=elapsed_time_formatted, calories_burnt=calories_burnt)


if __name__ == '__main__':
    app.run(debug=True)

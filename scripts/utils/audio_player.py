from pygame import mixer

# def play(file='./robot_comm_audio/front_front_fast.wav'):
#     mixer.music.load(file)  # Loading Music File
#     mixer.music.play()  # Playing Music with Pygame

def play(file='./robot_comm_audio/front_front_fast.wav', volume=1.0):
    if file[-8:] == 'beep.wav':
        # mixer.music.load(file)  # Loading Music File
        mixer.Channel(0).set_volume(volume)
        mixer.Channel(0).play(mixer.Sound(file))
    else:
        mixer.Channel(1).set_volume(volume)
        mixer.Channel(1).play(mixer.Sound(file))


    # mixer.music.play()  # Playing Music with Pygame

def stop():
    mixer.music.stop()



mixer.init() #Initialzing pyamge mixer

if __name__ == "__main__":
    # play(file='./robot_comm_audio/front_front_fast.wav')
    play(file='./robot_comm_audio/beep.wav', volume=0.1)
    input("press ENTER to stop playback")
    play(file='./robot_comm_audio/beep.wav', volume=0.2)
    input("press ENTER to stop playback")
    play(file='./robot_comm_audio/beep.wav', volume=0.3)
    input("press ENTER to stop playback")
    play(file='./robot_comm_audio/beep.wav', volume=0.7)
    input("press ENTER to stop playback")
    play(file='./robot_comm_audio/beep.wav', volume=1.0
         )
    input("press ENTER to stop playback")
    stop()

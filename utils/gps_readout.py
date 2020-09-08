import os
import time
import math
import threading
from gps import *

gpsd = None  # seting the global variable


class GpsPoller(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        global gpsd  # bring it in scope
        gpsd = gps(mode=WATCH_ENABLE)  # starting the stream of info
        self.current_value = None
        self.running = True  # setting the thread running to true

    def run(self):
        global gpsd
        while gpsp.running:
            gpsd.next()  # this will continue to loop and grab EACH set of gpsd info to clear the buffer


if __name__ == '__main__':
    gpsp = GpsPoller()  # create the thread
    try:
        gpsp.start()  # start it up
        while True:

            def readCoordinates():
                lat = gpsd.fix.latitude
                lon = gpsd.fix.longitude
                gps_time = str(gpsd.utc) + ' ' + str(gpsd.fix.time)
                speed = float("{0:.4f}".format(gpsd.fix.speed))
                alt = float("{0:.4f}".format(gpsd.fix.altitude))
                climb = float("{0:.4f}".format(gpsd.fix.climb))
                track = gpsd.fix.track
                sats = gpsd.satellites
                eps = gpsd.fix.eps
                epx = gpsd.fix.epx
                epv = gpsd.fix.epv
                ept = gpsd.fix.ept
                fixtype = gpsd.fix.mode

                if (math.isnan(lat)):
                    lat = "NAN"
                else:
                    lat = "%s " % lat

                if (math.isnan(lon)):
                    lon = "NAN"
                else:
                    lon = "%s " % lon

                if (math.isnan(speed)):
                    speed = "NAN"
                else:
                    speed = "%s km/h" % speed

                if (math.isnan(alt)):
                    alt = "NAN"
                else:
                    alt = "%s m" % alt

                if (math.isnan(climb)):
                    climb = "NAN"
                else:
                    climb = "%s m/s" % climb

                if (math.isnan(track)):
                    track = "NAN"
                else:
                    track = "%s" % track

                if (math.isnan(eps)):
                    eps = "NAN"
                else:
                    eps = "%s" % eps

                if (math.isnan(epx)):
                    epx = "NAN"
                else:
                    epx = "%s" % epx

                if (math.isnan(epv)):
                    epv = "NAN"
                else:
                    epv = "%s" % epv

                if (math.isnan(ept)):
                    ept = "NAN"
                else:
                    ept = "%s" % ept

                # sats_str = ','.join(sats)

                if fixtype == 1:
                    fixtype = "No Fix"
                else:
                    fixtype = "%sD" % fixtype

                coords = [gps_time, lat, lon, alt, speed, climb, track, eps, epx, epv, ept, fixtype]

                return coords


            coords = readCoordinates()
            d = open('gpsData.txt', 'a')
            d.write('\n%s' % ','.join(coords))

            print("\n")

            # gps_time = coords[0]
            # latitude = coords[1]
            # longitude = coords[2]
            # altitude = coords[3]
            # heading = coords[6]
            # speed = coords[4]
            # climb = coords[5]
            # fi = coords[12]

            print("gps time:  ", coords[0])
            print("Latitude:  ", coords[1])
            print("Longitude: ", coords[2])
            print("Elevation: ", coords[3])
            print("Heading:   ", coords[6])
            print("Speed:     ", coords[4])
            print("Climb:     ", coords[5])
            print("Fix:       ", coords[11])
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):  # when you press ctrl+c
        print("\nKilling Thread...")
        d.close()
        gpsp.running = False
        gpsp.join()  # wait for the thread to finish what it's doing
    print("Done.\nExiting.")

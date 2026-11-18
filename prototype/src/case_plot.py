"""Plot raw network conditions and the three paired playback trajectories."""
from PIL import Image, ImageDraw, ImageFont


def plot(out,case,trace,histories):
    img = Image.new("RGB",(1250,865),"white")
    d = ImageDraw.Draw(img)
    def font(size):
        try: return ImageFont.truetype("arial.ttf",size)
        except OSError: return ImageFont.load_default(size=size)
    colors = dict(fixed="#9b6024",buffer="#44759c",q="#8e367b")
    d.text((70,20),"Paired playback on one test trace",font=font(34),fill="#222831")
    d.text((70,68),case,font=font(19),fill="#454e58")
    for i,(p,c) in enumerate(colors.items()):
        x=700+i*150
        d.line((x,40,x+30,40),fill=c,width=4)
        d.text((x+37,28),dict(fixed="Fixed",buffer="Buffer",q="Q-learning")[p],font=font(18),fill=c)
    xmax=max(h[case][-1]["network_time_s"] for h in histories.values())
    t,b=trace
    panels=[("Trace throughput (Mbps)",max(b[1:])),("Selected bitrate (Mbps)",4.3),("Buffer at request (seconds)",max(s["buffer_before_s"] for h in histories.values() for s in h[case]))]
    for i,(label,ymax) in enumerate(panels):
        left,right=85,1190
        top,bottom=153+226*i,310+226*i
        def pixel(x,y): return (left+(right-left)*x/xmax,bottom-(bottom-top)*y/ymax)
        d.text((left,top-32),label,font=font(19),fill="#222831")
        for j in range(5):
            y=ymax*j/4
            _,yp=pixel(0,y)
            d.line((left,yp,right,yp),fill="#dddfe2")
            d.text((left-8,yp),f"{y:.1f}",anchor="rm",font=font(15),fill="#555")
        for j in range(6):
            x=xmax*j/5; xp,_=pixel(x,0)
            d.text((xp,bottom+5),f"{x:.0f}",anchor="mt",font=font(15),fill="#555")
        if i==0:
            pts=[]; offset=0.
            while offset<xmax:
                for j in range(1,len(t)):
                    if offset+t[j-1]>=xmax: break
                    pts.extend([pixel(offset+t[j-1],b[j]),pixel(min(xmax,offset+t[j]),b[j])])
                offset+=t[-1]
            d.line(pts,fill="#324f47",width=2)
        else:
            field=("bitrate_mbps","buffer_before_s")[i-1]
            for p,c in colors.items():
                pts=[]
                for row in histories[p][case]:
                    point=pixel(row["network_time_s"],row[field])
                    if i==1 and pts: pts.append((point[0],pts[-1][1]))
                    pts.append(point)
                d.line(pts,fill=c,width=3)
    d.text((620,815),"Elapsed network replay time (s, excluding simulated RTT)",anchor="mt",font=font(19),fill="#333")
    img.save(out/"case.png")

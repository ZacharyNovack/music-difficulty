//~ Revision: 371, Copyright (C) 2016-2020: Willem Vree, contributions Stéphane David, Pedro Martínez.
//~ This program is free software; you can redistribute it and/or modify it under the terms of the
//~ GNU General Public License as published by the Free Software Foundation; either version 2 of
//~ the License, or (at your option) any later version.
//~ This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
//~ without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//~ See the GNU General Public License for more details. <http://www.gnu.org/licenses/gpl.html>.

'use strict'
var follow_VERSION = 371;
var pre_opt = {}, abc_arr, demoDlg, media_file, abc_enc, offset_js, times_arr, opt, lpRec, maat_duren;
//var fs = require('fs');

(function () {
var ntsPos = {}, ntsSeq = [], iSeq = 0, isvgPrev = -1, ySvgs, gScale, gMidi, gOutput, gOutPorts = [], gChk = {};
var gAbcTxt, gToSynth = 0, isPlaying = 0, prevTime, prevDt, nVoices = 0, allNotes, menuVisible;
var gStaves, iSeqStart = 0, timingData, metDelay = 1, keyboardVisible = 0;
var deNot, hasSmooth, timer1, timer2, timer3, timer4, timer5, timer6, timerAbc, xScrollStart, startKnop, loopt;
var geelNeer = {}, neerTijd = {};       // midinum -> aanslagtijd
var gSpeelVol = 70, gFollowVol = 110;   // volume of play back and followed staff
var rMarks = [];        // a marker for each voice
var pMarks = {};        // permanente markeringen
var pMarksType = {};    // fout type van markering
var isvgPrev = [];      // svg index of each marker
var mrkVce;             // stem van de laatst gemarkeerde noot
var $loper = $(document.createElementNS ('http://www.w3.org/2000/svg','rect'));
$loper.attr ({ id: 'loper', fill: '#0dd', 'fill-opacity': 0.5, width: '0' });
var loper = $loper.get (0); // geen jquery met de loper
var atag, btag, selectedTag = null, lusStart, lusEnd;
var audioCtx;
var golven = [];
var liggend = [];
var deb, sustain = 0, sustained = {}, tmperr;
var midiLoaded = {};    // midi nums of already loaded waves
var midiUsed = {};      // midinumbers used
var ntsPos = {};
var lastBars = {};
var tmsSeq = [], ixTms;
var gTrans = [];        // playback transposition for each voice
var stfPos = [];
var stfHgt = [];
var allNotes = [];
var MidiOutput = [];
MidiOutput.push(["Start!",0].toString() + "\n");
var inFileName = '';
var errorCount, adaptiveMode = 0, goodNotes, lateNotes, errorMarks = {};
var elmSgd, elmSe1, elmSe2, elmSe3, elmPct, elmStot, perfElm;
var $error, $tempo, $mtrpad, kwartTempo, curNote, nuKwartTempo, nuTfac, korteNoten;
var E3 = "//PEZAAAAAGkAKAAAAAAA0gBQAAATEFNRTMuOTkuNVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVOLHpcKqmSyYMDLK//0bSOmcN3/zzuko9qY//6YkwnAm+x0pDgA1EizAzvKCAzVBebiwE+XDQDU0cgDFUKMDI2KEDHMB675TJ8d5oXwMwpCQMYA0gMTZCQMsYyQMOwgOzsbmBoVHNAMXISQMHwgAMNAHAMSATABAEgYFQBt+aOggs3NwMVQlgFBqAYSQWgYTwPgAAOAGAgBjtDL/1W/AwwC4Aw/AaAwDACAwMAkAkCADEoDIDA+C8GAHDPv/7N/wMEATAMIgDAMbgkAJB0AYAGAEE0DBqBsLXAMGQHAsNAUAeRf//7N9/4cQIYBg+BEDYyAoAshoGAEDIAQNwMGoC//PEZP8oJh0e3s3ZARrh4hS3jaAiQ+QMtidAIgFAOAuAEDcLOf////+7/bJYBgAApAOUAsAAMjk8A0BcbAgmFp4r4nMDAyAsMtBaYFzAZcE+AYEgAERPoAgAQQDcD9YXsLv3ErBN9xKCX7rTAyYkLdAHDPL5w0KgGMFAQGgaMWBphX0/FmkmM2UiIf/Ygo9jgE6Hk//rTTIITiZUFJixgBG/+hb8UkIMEeHpwokTJ/847z///zifcJ3af///SXcym+pKTcZCtakbKOhiiQFKBAUEDzGnDXCzqkzvHDMoD2LxRUak+BAhUJJljSUuM6IIAmFJES8wRFMqXFQKkCuFMhS4hCN5TZLrFlS5TInjfVTFR4xgVuMix6ymQjCBkmckOuYzaGkh1xvzBgcJfapWiNCL8vOyl2HKWiKMCrU1QMcHNDnA6oFUWUEqo0pRNIUYfZJpXK91MFSLEFhoSkoAAIudRoMsUCpC3N4HML9vCjJFZ2UsRDKDoCZMNlwQuRSp//PEZL0vwg1DC+1gAZdp2mntz4AAUM7+bYmUvM5TEEdxpiFK72cUitzIVmrCPM7rOhEd4hCwWNdAgUtnCSWf8YCgMQEPe+8P44KKJMO+nsl6nkPNS4MCFJoCUSU7lKL0MI5hQD8qwQan2jPOCYpIXmbmYSlvGRsWUAbm2Asoou7KXcJXPKGFZwMy9Ymbvy2hnF+QO8clhT1tadxhL7wG6j6TzasylbS3hxXgz6OM1aDJ2xOyzK0+TD4Bjy6qzbQ1DrzzLsLqbvFX+j6mDhtYjklaq7kJBQBIAwBLHk8SFKSjhqwdg53rov88GCpSWFOXwpR/J3SsPDlC0ikgyyBQC3wFAPQHk8X3zx+o1Plxzc903lQmioZsh+pEtG84RQte6iYJkoGo+i66ldIvllH//k4f4IVwiRQF8u2lAOIjQnCXNVDXYkJWYdpVElsRZpp9ihMJYQtIcTCiBsw6H8tO1fQCTgzNRzIiITh4FyeMhlC6BLBGTBMsKoHk0kS7KSMl//PEZEwYYf1LKz0mjpzJ8lgAy88wtlUVF2S70k03MiFPkpgiFK2REUnlkLEsQ2oj88k02TSWScwrFtEJUYlajUmirrxDukUpTImJFNMmzm+Uv3InFyZ79mtqcP9748+1I0WU/5RpWM7Tmzjvk72bfiO8uXn6duPmtLFs3PX5BdMWmaRdT5vjC0YK5WmFsUfuQOdK4AT7jTcREpBkqeijjsDP6rcDRlhFLGfpZToUAgqkWy0giJR9BsQVFDBlhHSgOhiinLsnFHiOdapOxd0aPVUrDns8zwTS+wzf//tim25XgNCd3/K5KozC4qpOFo1KrbHr9oDRQWP//QNT342LKonv1KjJcbcISPE6HS01qE0Va5j8GahxqC5BvKU1PRMCKsKBJOqZshQkwIoXC6FaKqLUZ9N7ETMiyMhFD4sUIiYljO2TLv2lvp0hde8wQ/SZnwowIfwpmDE+SgJMVKECAgMBAKqCnDsB4YOSoZg1CBtVyT2y/3shsXHZm870gqg1//PEZIAUHYtJHz0jXqMiFkwAy9MxI6T5/SpFhQox/Nek5FfgkJ7NpDTFJUNKae9Jq0UfpaYqx1SbgqQi8ThhTh0GCqKoKp7LdlmUOs1chsbNzrCMsRXjwFzIQ1pgZKKMiDxya5lWosETDFlfrmlhwmejRbKOCdVhwCYIksO5WYcC6by8F5eqGdw1+178IlbWqGxPHKRhTvnFvVDUIUHChBvGMswo6Zb4C72WSDKRv////2gsTilxVqi4rCrNoCn//SHVTEFNRTMuOTku++9SCtv1uu4bs428hEoxtidEBckCJZSt6FNyxn6TDQKGu66S8pLooqmrFPakdrb+ei5qWWdVpQ2eB5105tduSqi71DZ2dHmyKIcvcwcHDoBDaWpWtxtbpLLaHCJ7j1RU70opepW9SXc7e0DBUFcsDUNAG53iJBbsaghxTYfpmMEmNxf1CtSpUSiBWqRPGl8qFkqdq1TNVQERjqhxABIgK6lAAomQbbgKljldTaJwiLEXQVhW//PEZLQOsMNPLySrcipyKjgCxp69pL1UxwCXfpohIZEgrxgs7Eo0uYsuvB23GFic3BrvPzz/oIvy23BjzWnhTqHRii+M/D6FG4ewkQcw4xqgaSTJ0rWM7EON6CpVKbiKtCWf11Nv+VlmA2q5hbznWCoKpDr6YV1vMalGpHMPebkVtabvVTjkOaUeNDBposElSrkPnwlqU0+pq9bUHFfh2PuQQ+0PUsE1LeMii0gZ+5RZVf+dajgXhioO3P4eJiLKl6z1ATuXh/Qzk9MjHpmZnU8LT2ANNdrHMwH5iw46efR4odOvVXP0RzM/M5abWYcmSkMhKfWmsxJcelqOOnbWZLOdjp+SJjGk4lX8kSgdfQ+A2///01y2+cav/yKNX2qwYKWa15R2VWtSW+q51VLo5SX7HXhqLkcNR5zolEruSSLQlRshby1GnRHA6wQYCLsKnLhkQx/oKC4N9XAecsDgKfSRRbCgMykkiCvE10xqwGD0DjAkjmzD5DR4wxpoKHGs//PEZP8Xzec3C2mGnS5CijAA0wXE60TRYVJchqOpGwlhrualM0/KUsESBlyrI6/72ax/5LA1/JTSgYbGaVHowYCNS57H9sKdLSQHICXu6UD8Y9LmM5MIxE14poUoT0yzEPHz/WIoNy4hATGKRWYXwam2meOjll5JKvF4fyCmoX/4hZx6OzZI0cts4Ho96kxBTUUzLjk5LjWqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqiQC0+3bc0/ryzcOwXDtRuNhlMpnZPDNuWu6/0dRqYu4ietNbcUhihixJsx7abiurUgXHKXSuAHCr07x0KK5uj5kR/CITYu7/9OvEqtLaAYV/p5xomXgohURZxxqbyppy3+/wsw1EyBOD4RRsQnCKKdVLtmyhRj2nFBhB4x0f+oSmqK32/3OS6aX9M3VjXQ8fNdS2axhT/qxbhVv/yooaQCb5ESVvU88ttQiGn5nkFXdbA3+Mnf6QQJEXgCrnmX+iDB0NUtZ/FnF0nIOrgGJi02///PERNET2ckio2EnpyeDEkSmww8yMaoIzo7PzVs2WYYwbuLjJXJuMJ9HszOXlsspq9AFQTWd6WENOJyOpgSefYItauDxxap36LqIQySKlA9rmzjDuY651nLDjTDHik/fqnGC0ueREaNVbVyhwrKDJY0tufyJ1ht+2+S/j2//KihqAfRH2HopbsrxQwau6wKHKmLzz0yqUhCKTeRNZQ8GEzMgSIoFU6/UD2BkRMzDMwaVrJh1Z1GZsIKlUCLGeZopvHsIeqMn8BYJikSbCwxcaXi3vS6i8H6ptigOSf9rRmoUpyrE7GKYXB3RUQYy6eLhvZk4PFXytgxVCiezzv4a5amzw9eSTP8qQQJ/NsUjzENJraO2Mkv+uqmd5/l6CQq0ZmONv96ORwTGZWDIqphd8VnlqvYjO+sniv2x52GhwQjGOg0iVjA6MtokAJdQ+FgaLqvV2JJUiIRgALMkhBY8VjLYKDCqhbQCHiqExMkEiI29UJoniWsn8X+FBLDUTi5L//PERP8WwUsOAGninC4jEhwA08vEAoxJIZncZVAA6IZM2GCkY9vGx5+P/4851+9eicP/SrsLAd3ZyTXKzR4i1FMVfrSBZE3ldo0Rl9RdPdSR2p617h63AbnLeVyOVNrMrEHo/VyicWhwFzrdpHyql//CLdLuwtv5Xoi2bc33yDHon9FxGkwn4E6aBr9/VxI5kkNl+WEp2v/CFdIco2/bapkDAiGA4kAmGAghAka0hS+A6vGKiayDAwgxBqMWHUJarqooCLAq0L5RVk8SeUSCrhc3xLlrNUaLiaygNETZ8oA8M0b+AcMsj9RPmtHGgLkoLqaPFRLKiki4PEerI49kLW2xQt7g8Y4Lp7mVwfyST/ba15OVErkhzs51yyxo7Y7m8KA5Rpfmfql6FM4RdL7orPJtV7/1Et5PwUJwGBJMy57qzuF7V/szFj1/gaCP3YGQJO+Wvk1FJdFhAcmaoGIwUWGaJXIYRJ2ApGMHBjZjMLhT+QthYqBxM8i+BKnyeaAN//PERPoWcWcKAG3lqix7FhQA28U4ZtOBLx4rcvjpJk0HMjQXsE71Zd1/DOKSRFluUKqhH0EcVcBeYoxsrYk56MQ+Rdn6hMwebYpE+5KFabZpYXlcffvvWRgMdabF4VqueQ40zdl7TwYTuC8+Z+Vwu7PBPf7u8ysV9jt/F++3rxSVOyAOHXCfaT3GZtbXI7EpliwK/2zKMLSWqwZf4hAhIAzD4CDguYHBANDSBy2W3MRjYwQGTCwDBxMMLmUwgER4dsWXIsZvGrLGQ5IIpXXMBAGSELPo8Isz8S4YxxmQdYbOUUbThv+Ar29uPJUvmt4XIIDNiDtSZ0zH4cUAYyWeroZySTDjXKii6lxvTqTGI3gxW0x2JcGGRRJlQeudvO2uvD000ieV3hTUoVCkPp391L//gCItkuo2AHwGCs1gKYtVXnIApdbjL3YGKgk7Aqy3dXXK3Wa4MlAcHKvEAgQgbuuwn2YUWshAwOYgGGSixhAQ0yLP06i71msSbK8FDUS1//PERP8W3WsEAHHlqi8TEgwG28vGfWckUR3fg1p7iRxnUMkwFKm6sUnJ3/uyypWfFrr7Vn6YuhVJuwThK2w22GPO1BdJEIy27UUKm68L3RWyyP/nTK4fMXwoxMi9OEdgI9U2YZ9NrbCi+A3Rsz/O9kMy1JcSurdi8RsLPQM/5Ct9/6jjJb////cqTEFNRTMuOTkuNaqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqb/kXv3J+D2n6WePHsYbsHl3TKlo2LF5G9D24nHShMF7kmXZjjriqVIDOhQXNLkKFl/Vxsa0cBb+xoixEK43Caqs/L9m/erlQ8juzqraEvyUOP67v1Dv+/+h3R/pPEUvx/GtUs38HbTRhUhie18tXuQ+xt+NZPWhxgbBpjMpNPUoqGmCljLgYAq5T//PEZIoLiQ0oy02HGyZjKiCuyw8ymUy2vSK4V5I/ScUrRN9MxEFuJona0jmZnFW6zu3daBZrtpZCrEPSEXzB12D0tl9xOyKvVfqLyIqEKiwHwnNlVLZqu3nCapWUdIoejc6qb31b/R//+h0x2KtdjRUOBlkWMDm/WbCKBxM0LfYiTEFNRTMuOTkuNaqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqMjkUNSO7f8KkyE2S0tyZVivRopSXQGhTEhUhrKy7MrjXjaFIEjX//ppWN6hxW8rShnYym/ytlBCipejyoBI5S6sunmXcqS0CsGFKxqM5dAJ1IWRStJ30SS6VsMi1wsgexC7hVLmpVeu9iV0q2osru+oamEzp//PEZHUOIQ0hHz0iLhvZ7jl2YMsmXYCpYRlkySmBaQj0+Hk4LS3zlx2Werj37ZkxWrUwlPNdbe6sPUhxMpeTKURFUf63LMdjAKKm/apXob0KyG9TfmeVlDrWU+e//LNf//HNNCqQKKmRjTIsSNCyB7DAVImRZIxpoc00KpAoqZGVTEFNRTMuOTkuNVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUNrmJJJLQwNA8o//PEZAoFmAb2fwwjARIh1dGeMMSeWFU1inUzqFtbUf+sU/xfrb/9Yp/+tv/7P/xUW/+sUb/8VFgCV5cdsFABhJlxbOUVcJ48oIFUMmWUGEFkZNZ2UjWWzI+hgYME4g+wHY5DlMqIrfgjjOVP//ymVUXMFDKLB40FRRtQsK///8UVTEFNRTMuOTkuNVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
var metroGolf, gCountIn = 0, metroIsLow = 1, checkNotes, timeNotes = 0, cursorNotes, colorNotes = {}, numNotes;
var neerNt = {};    // midinummer -> noot record uit ntsSeq
var metTimeout = 0, minTempo = 0, tfacmin, fHoog, fLaag, topSpace = 500, dottedHeight = 30, keyTransparency = 0;
var tmap = {}, tmapI = {}, toetsNeer = {}, deSvgs = [], deSvgGs = [], keyMark = 0, blackKeys, txtmapI = {};
var extrStf = 0, scoreOrig, scoreIx = 0, curStaff = 0, prevPlayRest, sok, isMaster = 0;
var ntsTimes = {}, slaveTime, gCountInPb = 0, gAbcSave, barTimes, chkModDef = '2', opt, micDlgVisible = 0, minLevel;
var micNote = 0;
var deStroom, bronKnoop, analyser, frame, gainKnoop, graaf_breedte, frame_len, diffs, tijdMeter, maxDt = 0, lusTel = 0, drempel;
var graaf, graaf_breedte = 512, levMeter, levMax = 0, uitslag = -100, nootElem, centMeter, centSpace, centAvg = 0, centLen = 30;
var frameMax, tmrID, rafID;
var noteStrings = ["C ","C#","D ","D#","E ","F ","F#","G ","G#","A ","A#","B "], stepnm, accnm;
var centArr = [];
for (var i = 0; i < centLen; i++) centArr.push (0);
var opt_url = {};
var optMenu = { delay:1, eenvoor:0, metro:1, keys:0, mark:0, mute:1, volgmod:6, chkmod:2, extract:1, 
    portsel:'synth', tempo:100, kbopa:0, pw:0, scl:1, gain:5, drempel:1, minlev:-18, bass:0, mtpo:20, hrz:0, 
    vol:50, nocur:0, rstmute:1, exact:0, nolyr:0, lus:0, lusstop:1, lncur: 0, tmng: 0, tmng2: 0,
    metmrk:0, ftyp:0, nocnt:0, scal:0, echo:1 };
var optRest = { staff:1, line:30, begin:0, btime:0, etime:0, // hebben (nog) geen menu equivalent
    ctrls:0, shrnk:0, lat:0, pnlerr: 0.5, pnltim: 0.5, oneclr: 0, late1: 150, late2: 1000,
    pnlkrt: 0.5, kort: 30, minkort: 200,
    atagTijd:-1, btagTijd:-1, audOn:true, hide2cur:0, stopat:-1, hinfo:36, mat:0.5, // opacity
    tmo:200, kleur: ['#f9f','#3cf','#c99','#f66','#fc0','#cc0','#ccc'], // wachtende wijzerkleur per stem
    ntGoed:'#080', ntFout:'#f00', ntWacht:'#0dd',   // wijzerkleur als gespeeld
    ntKleur: ['#f00','#ff0','#f00','#0bb','#f00','#aaa','#f00']};       // permanente foutmarkeringen
var masterVol = 2, gGolvenKlaar, latency, nAsyncLoads = 0;
var audElm, volElm, play_start = 0.1, play_end, audSpeed, mediaFnm, audSeq; // audio stuff
var hasAud;     // after canplaythrough event is received
var hasAudUrl;  // the audio player has a source URL
var gCurMask = 0;   // wijzermasker
var prevKeys = {};  // stemnummer -> [toetsRect, nootNaamElem]
var exactEnabled = 1;   // only false during playback mode
var kleuren;    // kleuren van de permanente markeringen
var startTime;

function neer (midiNum, vol, time) {
    var source = audioCtx.createBufferSource ();
    source.buffer = golven [midiNum];
    var gainNode = audioCtx.createGain();
    gainNode.gain.setValueAtTime (0.00001, time);   // begin bij -100 dB
    var vol = masterVol * vol / 150;
    if (vol == 0) vol = 0.00001;    // stem kan volume 0 hebben.
    gainNode.gain.exponentialRampToValueAtTime (vol, time + 0.003);
    source.connect (gainNode);
    gainNode.connect (audioCtx.destination); // verbind source met de sound kaart
    source.start (time, 0.025);  // start time, offset in buffer (in seconds)
    liggend [midiNum] = [source, gainNode, vol];
    delete sustained [midiNum];
}

function op (midiNum, time) {
    var tup = liggend [midiNum];
    if (tup) {
        if (sustain) {
            sustained [midiNum] = 1;
        } else {
            var [source, g, velo] = tup;
            g.gain.setValueAtTime (velo, time); // begin release at current volume
            g.gain.exponentialRampToValueAtTime (0.00001, time + 0.1); // -100 dB
            source.stop (time + 0.1);           // stop na uitsterven
            liggend [midiNum] = undefined;
        }
    }
}

function metroNeer (luid) {
    var source = audioCtx.createBufferSource ();
    source.buffer = metroGolf;    
    var gainNode = audioCtx.createGain();
    gainNode.gain.setValueAtTime (luid ? 3.0 : 1.0, 0);
    source.connect (gainNode);
    gainNode.connect (audioCtx.destination); // verbind source met de sound kaart
    source.start (0, 0.06, 0.16); // de mp3 file heeft 0.06 sec preamble en het geluid duurt 0.16 sec
}

function laadNoot (midiNums) {
    var notes = 'C Db D Eb E F Gb G Ab A Bb B'.split (' ');
    var ix = midiNums.shift (); // midiNums wordt opgegeten
    if (!ix) {                  // alle noten zijn geladen
        gGolvenKlaar = 1;
        logerr ('midi nrs: ' + tmperr.join (','));
        $('#comp').toggle (false);
        return;
    }
    var noot = notes [ix % 12], oct = Math.floor (ix / 12) - 1;
    var xs = MIDI.Soundfont.acoustic_grand_piano [noot + oct].split (',')[1];
    decodeer (xs, function (buffer) {
        golven [ix] = buffer;
        $('#comp').append (', ' + ix);
        tmperr.push ('' + ix);
        midiLoaded [ix] = 1;    // onthoud dat de noot geladen is
        laadNoot (midiNums);    // laad de volgende noot
    });
}

function laadNootHulp (midiUsed) {
    gGolvenKlaar = 0
    console.log(midiUsed)
    var midiNums = Object.keys (midiUsed).filter (function (m) { return !(m in midiLoaded); });
    if (gToSynth == 0) midiNums = [];   // skip loading if synth not used
    if (midiNums.length) {
        //~ console.log ('audioCtx.state: ', audioCtx.state);
        $('#comp').toggle (true);
        tmperr = [];
        laadNoot (midiNums);
    } else {
        gGolvenKlaar = 1;   // noten al geladen
    }
}

function readDbxFile (files) {
    var url = files[0].link;
    url = url.replace ('www.dropbox', 'dl.dropboxusercontent').split ('?')[0];  // make direct link
    //~ scoreFnm = files[0].name.split ('.')[0];
    pre_opt = {};   // clear old preload settings
    initOptions ();
    waitDlg ('#wait', 1);
    logerr ('link: ' + url);
    $.get (url, '', null, 'text').done (function (data, status) {
        logerr ('preload: ' + status);
        readAbcOrXML (data);
        waitDlg ('#wait', 0);
    }).fail (function (jqxhr, settings, exception) {    // same origin policy
        $('#wait').append ('\npreload failed: ' + settings);
    });
}

function readAbcOrXML (abctxt) {
    var xs = abctxt.slice (0, 4000);    // only look at the beginning of the file
    if (xs.indexOf ('abc_arr = [') >= 0) {  // its a preload
        const encodedJs = encodeURIComponent (abctxt);
        const dataUri = 'data:text/javascript;charset=utf-8,' + encodedJs;
        nAsyncLoads += 1;
        getPreload (dataUri);
        return;
    }
    if (xs.indexOf ('X:') >= 0)      { dolayout (abctxt); setUrlOptions (); return }
    if (xs.indexOf ('<?xml ') == -1) { alert ('not an xml file nor an abc file'); return }
    var xmldata = $.parseXML (abctxt);
    //console.log(xmldata);
    var options = { p:'f', t:1, u:0, v:3, m:2, mnum:0 }; // t==1 -> tab en perc naar %%map
    var res = vertaal (xmldata, options);
    // console.log(res);
    if (res[1]) logerr (res[1]);
    dolayout (res[0]);
    setUrlOptions ();
}

function readLocalFile () {
    var freader = new FileReader ();
    freader.onload = function (e) { 
        readAbcOrXML (freader.result);
        waitDlg ('#wait', 0);
    }
    var f = $('#fknp').prop ('files')[0];
    inFileName = f.name.split ('.')[0];
    if (!f) return;
        //~ scoreFnm = f.name.split ('.')[0];
    var nm = f.name.toLowerCase ();
    waitDlg ('#wait', 1);
    if (nm.endsWith ('.ogg') || nm.endsWith ('.mp3')) {  // its an audio file
        media_file = window.URL.createObjectURL (f);
        addMediaFile ();
        waitDlg ('#wait', 0);
    } else {    // .xml, .abc or preload
        pre_opt = {};   // clear old preload settings
        initOptions ();
        freader.readAsText (f);
    }
}

function dolayout (abctxt, iSeq_p) {
    var percSvg = ['%%beginsvg\n<defs>',
        '<text id="x" x="-3" y="0">&#xe263;</text>',
        '<text id="x-" x="-3" y="0">&#xe263;</text>',
        '<text id="x+" x="-3" y="0">&#xe263;</text>',
        '<text id="normal" x="-3.7" y="0">&#xe0a3;</text>',
        '<text id="normal-" x="-3.7" y="0">&#xe0a3;</text>',
        '<text id="normal+" x="-3.7" y="0">&#xe0a4;</text>',
        '<g id="circle-x"><text x="-3" y="0">&#xe263;</text><circle r="4" class="stroke"></circle></g>',
        '<g id="circle-x-"><text x="-3" y="0">&#xe263;</text><circle r="4" class="stroke"></circle></g>',
        '<path id="triangle" d="m-4 -3.2l4 6.4 4 -6.4z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="triangle-" d="m-4 -3.2l4 6.4 4 -6.4z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="triangle+" d="m-4 -3.2l4 6.4 4 -6.4z" class="stroke" style="fill:#000"></path>',
        '<path id="square" d="m-3.5 3l0 -6.2 7.2 0 0 6.2z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="square-" d="m-3.5 3l0 -6.2 7.2 0 0 6.2z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="square+" d="m-3.5 3l0 -6.2 7.2 0 0 6.2z" class="stroke" style="fill:#000"></path>',
        '<path id="diamond" d="m0 -3l4.2 3.2 -4.2 3.2 -4.2 -3.2z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="diamond-" d="m0 -3l4.2 3.2 -4.2 3.2 -4.2 -3.2z" class="stroke" style="stroke-width:1.4"></path>',
        '<path id="diamond+" d="m0 -3l4.2 3.2 -4.2 3.2 -4.2 -3.2z" class="stroke" style="fill:#000"></path>',
        '</defs>\n%%endsvg'];

    function perc2map (abcIn) {
        var fillmap = {'diamond':1, 'triangle':1, 'square':1, 'normal':1};
        var abc = percSvg, ls, i, x, r, id='default', maps = {'default':[]}, dmaps = {'default':[]};
        ls = abcIn.split ('\n');
        for (i = 0; i < ls.length; ++i) {
            x = ls [i];
            if (x.indexOf ('I:percmap') >= 0) {
                x = x.split (' ').map (function (x) { return x.trim (); });
                var kop = x[4];
                if (kop in fillmap) kop = kop + '+' + ',' + kop;
                x = '%%map perc'+id+ ' ' +x[1]+' print=' +x[2]+ ' midi=' +x[3]+ ' heads=' + kop;
                maps [id].push (x);
            }
            if (x.indexOf ('%%MIDI') >= 0) dmaps [id].push (x);
            if (x.indexOf ('V:') >= 0) {
                r = x.match (/V:\s*(\S+)/);
                if (r) {
                    id = r[1];
                    if (!(id in maps)) { maps [id] = []; dmaps [id] = []; }
                }
            }
        }
        var ids = Object.keys (maps).sort ();
        for (i = 0; i < ids.length; ++i) abc = abc.concat (maps [ids [i]]);
        id = 'default';
        for (i = 0; i < ls.length; ++i) {
            x = ls [i];
            if (x.indexOf ('I:percmap') >= 0) continue;
            if (x.indexOf ('%%MIDI') >= 0) continue;
            if (x.indexOf ('V:') >= 0 || x.indexOf ('K:') >= 0) {
                r = x.match (/V:\s*(\S+)/);
                if (r) id = r[1];
                abc.push (x);
                if (id in dmaps && dmaps [id].length) { abc = abc.concat (dmaps [id]); delete dmaps [id]; }
                if (x.indexOf ('perc') >= 0 && x.indexOf ('map=') == -1) x += ' map=perc';
                if (x.indexOf ('map=perc') >= 0 && maps [id].length > 0) abc.push ('%%voicemap perc' + id);
                if (x.indexOf ('map=off') >= 0) abc.push ('%%voicemap');
            }
            else abc.push (x);
        }
        return abc.join ('\n');
    }
    gAbcSave = abctxt;  // bewaar abc zonder wijzigingen
    var xs = abctxt.split ('\n');
    if (opt.nolyr) xs.splice (1, 0, '%%writefields w ' + !opt.nolyr)
    if (opt.hrz > 0) xs.splice (0, 0, '%%singleline 1', '%%maxshrink ' + opt.shrnk);
    else xs.splice (0, 0, '%%measurenb 0');
    if (opt.pw || opt.hrz) {
        var ys = [], i, x;
        for (i = 0; i < xs.length; ++i) {
            x = xs [i];
            if (x.indexOf ('$') >= 0) x = x.replace (/\$/g, '');            // remove all $
            if (x.indexOf ('I:linebreak') == 0) x = xs[i];                  // but keep I:linebreak $
            if (x.indexOf ('V:') == 0) x = x.replace (/snm="[^"]*"/ ,'')    // verwijder alle korte instument namen
            if (x.indexOf ('%%page') == 0) continue;    // verwijder page format
            if (x.indexOf ('%%left') == 0) continue;
            if (x.indexOf ('%%right') == 0) continue;
            if (x.indexOf ('%%scale') == 0 && opt.hrz) continue;    // verwijder ook schaal als horizontale layout
            ys.push (x);
        }
        xs = ys;
        if (opt.hrz) xs.splice (1, 0, '%%scale ' + opt.scl);
        else {
            var w = opt.pw, l = (0.03 * w).toFixed (1), r = (0.01 * w).toFixed (1);
            xs.splice (1, 0, '%%pagewidth '+w+'cm', '%%leftmargin '+l+'cm', '%%rightmargin '+r+'cm');
        }
    }
    abctxt = xs.join ('\n');
    if (abctxt.indexOf ('percmap') >= 0) abctxt = perc2map (abctxt);
    doModel (abctxt);
    doLayout (abctxt, iSeq_p);
}

function doModel (abctxt) {
    var abc2svg;
    var errtxt = '';
    var BAR = 0, GRACE = 4, KEY = 5, METER = 6, NOTE = 8, REST = 10, TEMPO = 14, BASE_LEN = 1536;
    var keySteps = [3,0,4,1,5,2,6];     // step values of the cycle of fifth
    var scaleSteps = [0,2,4,5,7,9,11];  // step values of the scale of C
    gAbcTxt = abctxt;
    allNotes = [];
    gTrans = [];
    var stfLns = [];    // number of lines for each staff

    function getStaves (voice_tb) {
        var xs = [];
        voice_tb.forEach (function (v, i) {
            if (xs [v.st]) xs [v.st].push (i); 
            else xs [v.st] = [i];
            if (v.clef.clef_octave) gTrans [i] = v.clef.clef_octave;
            stfHgt [v.st] = (v.stafflines || '|||||').length * 6 * (v.staffscale || 1);
            stfLns [v.st] = (v.stafflines || '|||||').length;
        });
        return xs;
    }

    function errmsg (txt, line, col) {
        errtxt += txt + '\n';
    }

    function parseModel (ts_p, voice_tb, music_types) {
        function setKey (v, sharpness) {    // voice, index in cycle of fifth (keySteps)
            acctab [v] = [0,0,0,0,0,0,0];   // step modifications for the current key in voice v
            alts [v] = {};                  // reset alterations
            curKey [v] = sharpness;
            var sign = sharpness >= 0;
            var accs = sign ? keySteps.slice (0, sharpness) : keySteps.slice (sharpness);   // steps modified by key
            accs.forEach (function (iacc) { acctab [v][iacc] += sign ? 1 : -1; });          // perform modification in acctab
        }
        var acctab = {}, accTrans = {'-2':-2,'-1':-1,0:0,1:1,2:2,3:0}, alts = {}, curKey = {}, tied = {};
        var i, n, p, oct, step, mn, noten, noot, fret, tuning, v, accidental, ntnm;
        var maatsoort = voice_tb [0].meter.a_meter;
        var tempo = 60; // quarters per minute
        var dtmp = 384; // duration of (metronome) beat in abc units
        maatsoort = maatsoort.length ? maatsoort [0] : {top: '4', 'bot': 4};
        for (v = 0; v < voice_tb.length; ++v) {
            var key = voice_tb [v].key.k_sf;
            setKey (v, key);
            tied [v] = {};
        }
        nVoices = voice_tb.length;
        gStaves = getStaves (voice_tb);
        var hasScoreTempo = 0
        for (var ts = ts_p; ts; ts = ts.ts_next) {
            noten = [];
            switch (ts.type) {
            case TEMPO:
                dtmp = ts.tempo_notes.reduce (function (sum, x) { return sum + x; });   // tempo_notes = lijst met ABC-duren
                tempo = ts.tempo * dtmp / 384;          // BPM van een kwartnoot
                if (ts.time == 0) hasScoreTempo = 1;
                if (ts.time == 0 && !pre_opt.tempo) {   // als tempo niet in de preload
                    pre_opt.tempo = tempo;  // -> tempo voor metronoom in de menubalk
                }
                break;
            case REST:
                if (!hasScoreTempo && ts.time == 0) {   // eeste noot na kopdeclaratie zonder tempo (Q:)
                    if (pre_opt.tempo) tempo = pre_opt.tempo;
                    if (!pre_opt.tempo && !times_arr && !maat_duren) pre_opt.tempo = tempo; // default tempo als geen sync data
                }
                noot = { t: ts.time, nt: -1, dur: ts.dur, nm: '', ntkb: -1 };
                noten.push (noot);
                allNotes.push ({ t: ts.time, ix: ts.istart, v: ts.v, dur: ts.dur, ns: noten, tmp: tempo, mtr: maatsoort, met: dtmp, st: ts.st });
                break
            case NOTE:
                if (!hasScoreTempo && ts.time == 0) {   // eeste noot na kopdeclaratie zonder tempo (Q:)
                    if (pre_opt.tempo) tempo = pre_opt.tempo;
                    if (!pre_opt.tempo && !times_arr && !maat_duren) pre_opt.tempo = tempo; // default tempo als geen sync data
                }
                for (i = 0; i < ts.notes.length; ++i) { // parse all notes (chords)
                    n = ts.notes [i];
                    p = n.pit + 19;             // C -> 35 == 5 * 7, global step
                    v = ts.v;                   // voice number 0..
                    if (gTrans [v]) p += gTrans [v];    // octaaf transpositie in sleutel
                    oct = Math.floor (p / 7);   // C -> 5
                    step = p % 7;               // C -> 0
                    if (n.acc != undefined) alts [v][p] = accTrans [n.acc]; // wijzig acctab voor stap p in stem ts.v
                    accidental = p in alts [v] ? alts [v][p] : acctab [v][step];
                    ntnm = [step, accidental];  // nootnaam voor de toetsen
                    mn = oct * 12 + scaleSteps [step] + accidental;
                    midiUsed [mn] = 1;          // collect all used midinumbers
                    noot = { t: ts.time, nt: mn, dur: ts.dur, nm: ntnm, ntkb: mn };
                    if (p in tied [v]) {
                        tied [v][p].dur += ts.dur;      // verleng duur van vorige noot
                        if (!n.tie_ty) delete tied [v][p]; // geen verdere ties
                        noot.nt = -1;       // noot alleen behandelen als rust
                    } else if (n.tie_ty) {
                        tied [v][p] = noot; // bewaar ref naar r om later de duur te verlengen
                    }
                    if (ts.ti2 &&       // tie na tweede herhaling voor deze noot/akkoord
                        ts.ti2.notes.filter (x => x.midi == mn).length) // filter tied-noten uit akkoord
                        noot.nt = -1;   // noot behandelen als rust
                    noten.push (noot);
                }
                for (i=24; i < 108; i++){
                    midiUsed[i] = 1;
                }

                if (noten.length == 0) break;           // door ties geen noten meer over
                allNotes.push ({ t: ts.time, ix: ts.istart, v: ts.v, dur: ts.dur, ns: noten, tmp: tempo, mtr: maatsoort, met: dtmp, st: ts.st });
                break;
            case KEY: setKey (ts.v, ts.k_sf); break;    // set acctab to new key
            case BAR:
                setKey (ts.v, curKey [ts.v]);           // reset acctab to current key
                allNotes.push ({ t: ts.time, ix: ts.istart, v: ts.v, bt: ts.bar_type, tx: ts.text });
                break;
            case METER:                         // ritme verandering: nog te doen !
                maatsoort = ts.a_meter [0];
                break;
            }
        }
        rMarks.forEach (function (mark) {   // verwijder oude markeringen
            var pn = mark.parentNode;
            if (pn) pn.removeChild (mark);
        });
        isvgPrev = [];                      // clear svg indexes
        for (var i = 0; i < nVoices; ++i) { // een markering voor iedere stem
            var alpha = 1 << i & gCurMask ? '0' : ''
            var rMark = document.createElementNS ('http://www.w3.org/2000/svg','rect');
            rMark.setAttribute ('fill', opt.kleur [i % opt.kleur.length] + alpha);
            rMark.setAttribute ('fill-opacity', opt.mat);
            rMark.setAttribute ('width', '0');  // omdat <rect> geen standaard HTML element is werkt rMark.width = 0 niet.
            rMarks.push (rMark);
            isvgPrev.push (-1);
            prevKeys [i] = [];  // schermtoetsenbord, liggende stemmen
        }
    }

    var user = {
        'img_out': null, // img_out,
        'errmsg': errmsg,
        'read_file': function (x) { return ''; },   // %%abc-include, unused
        'anno_start': null, // svgInfo,
        'get_abcmodel': parseModel
    }
    abc2svg = new Abc (user);
    abc2svg.tosvg ('play', '%%play');   // houdt rekening met transpose= in K: of V:
    abc2svg.tosvg ('abc2svg', abctxt);
    if (errtxt == '') errtxt = 'no error';
    logerr (errtxt);
}

function doLayout (abctxt, iSeq_p) {
    var abc2svg;
    var muziek = '';
    var errtxt = '';
    var nSvg = 0;
    iSeq = iSeq_p || 0;
    isvgPrev [0] = -1;
    iSeqStart = iSeq_p || 0;
    ntsPos = {};    // {abc_char_pos -> nSvg, x, y, w, h}
    stfPos = [];    // [stfys for each svg]
    lastBars = {};  // ABC tekenpositie van de laatste maatstreep voor ieder systeem
    var stfys = {}; // y coors of the bar lines in a staff
    var xleft, xright, xleftmin = 1000, xrightmax = 0, lastbar = 0;

    function errmsg (txt, line, col) {
        errtxt += txt + '\n';
    }

    function img_out (str) {
        if (str.indexOf ('<svg') != -1) {
            stfPos [nSvg] = Object.keys (stfys);
            stfys = {}
            nSvg += 1;
            if (xleft < xleftmin) xleftmin = xleft;
            if (xright > xrightmax) xrightmax = xright;
            if (lastbar) lastBars [lastbar] = 1;
            lastbar = 0;
        }
        muziek += str;
    }

    function svgInfo (type, s1, s2, x, y, w, h) {
        if (type == 'note' || type == 'rest') {
            x = abc2svg.ax (x).toFixed (2);
            y = abc2svg.ay (y).toFixed (2);
            h = abc2svg.ah (h);
            ntsPos [s1] = [nSvg, x, y, w, h];
        }
        if (type == 'bar') {
            y = abc2svg.ay (y);
            h = abc2svg.ah (h);
            y = Math.round (y + h);
            stfys [y] = 1;
            xright = abc2svg.ax (x) + w;
            xleft = Math.round (abc2svg.ax (0));
            if (s1 in ntsPos) return // left repeat at end of line -> two consecutive bar symbols
            ntsPos [s1] = [nSvg, xright.toFixed (2), y, w, h];  // voor de lijnloper
            lastbar = s1;
        }
    }

    function getNote (event) {
        function moveTag (i) {
            var tag = null;
            if (selectedTag == atag) {
                if (iSeq == lusStart) iSeq = i; // houd cursor op eerste noot na verplaatsing
                lusStart = i;
                tag = atag;
                opt.atagTijd = ntsSeq [i].t
                if (i >= lusEnd) { iSeq = i; opt.atagTijd = -1; }   // forceer opnieuw beginplaatsing 
            }
            if (selectedTag == btag) { 
                lusEnd = i;
                tag = btag; 
                opt.btagTijd = ntsSeq [i].t
                if (i <= lusStart) { iSeq = i; opt.atagTijd = -1; } // beginplaatsing
            }
            if (tag == null) return;
            tag.classList.remove ('tagselect'); // reset selection
            selectedTag = null;
            drawTags ();
        }
        var p, isvg, x, y, w, h, xp, jsvg, i, ys, yp, yoff, ix, v;
        event.stopPropagation ();
        xp = (event.clientX + deNot.scrollLeft) * gScale;
        if (xp < xleftmin + 24 || xp > xrightmax) {  // click in the margin
            if (menuVisible) toggleMenu ();
            else toggleStart (iSeq);
            return;
        }
        yoff = $('#notation').position ().top - $('#notation').scrollTop ();
        jsvg = deSvgs.get ().indexOf (this);
        yp = (event.clientY - yoff - ySvgs [jsvg]) * gScale;
        ys = stfPos [jsvg];
        for (i = 0; i < ys.length; i++) {
            if (selectedTag) break;                 // geen balk selectie als we de lusmerkjes verschuiven
            if (ys [i] > yp) {                      // op staff i is geklikt
                if (i != curStaff) setVoices (extrStf ? curStaff : i);    // setVoices sets curStaff
                alignSystem (true);
                break;
            }
        }
        for (v = 0; v < nVoices; ++v) toetsVeeg (v);    // verwijder toetsMarkeringen
        for (i = 0, ix = 0; i < ntsSeq.length; ++i) {   // vind de noot waar geklikt werd
            p = ntsSeq [i].xy;
            if (!p) continue;   // onzichtbare rust
            isvg = p[0]; x = p[1]; y = p[2]; w = p[3]; h = p[4];
            if (isvg < jsvg && ix == 0) continue;
            ix = 1;             // juiste regel bereikt
            if (isvg > jsvg || i == ntsSeq.length - 1) { ix = i - 1; break } // einde van regel
            if (isvg == jsvg && xp < parseFloat (x) + w) { ix = i; break; } // rechterkant nootdoos voorbij x-klik
        }   // deze lus wordt altijd verlaten via de break => ix altijd geldig
        if (opt.lus && selectedTag) moveTag (ix);
        else {
            toggleStart (ix);   // stop bij klik, => iSeq = iSeqStart = ix en plaatsLoper
        }
        clearTiming ();
    }

    function getStfLayout () {
        var xs = abctxt.split ('\n');
        for (var i = 0; i < xs.length; ++i) {
            if (xs[i].search (/^%%score/) >= 0 || xs[i].search (/^I:score/) >= 0) break;
        }
        if (extrStf && i == xs.length) {
            alert ('Staff extraction only works with ABC files that have a %%score declaration');
            extrStf = 0;    // hier moet een %%score geconstrueerd worden
        }
        if (extrStf) {
            scoreOrig = xs [i];
            scoreIx = i;
            var iv = gStaves [curStaff].map (function (i) { return i + 1; });
            var stf = '(' + iv.join (' ') + ')';    // dit werkt alleen als alle voice ID's integers zijn
            if (extrStf == 2 && curStaff + 1 in gStaves) {
                var iw = gStaves [curStaff + 1].map (function (i) { return i + 1; });
                stf = '{' + stf + '(' + iw.join (' ') + ')' + '}'
            }
            var sc = '%%score'
            xs [i] = sc + Array (xs [i].length - stf.length - sc.length + 1).join (' ') + stf;
        } else if (scoreOrig) {
            xs [scoreIx] = scoreOrig;
        }
        abctxt = xs.join ('\n');
    }

    if (!abctxt) return;
    getStfLayout ();

    var user = {
        'img_out': img_out,
        'errmsg': errmsg,
        'read_file': function (x) { return ''; },   // %%abc-include, unused
        'anno_start': svgInfo,
        'get_abcmodel': null
    }
    if (opt.hrz == 0) user.imagesize = 'width="100%"';
    abc2svg = new Abc (user);
    abc2svg.tosvg ('abc2svg', abctxt);
    if (errtxt == '') errtxt = 'no error';
    logerr (errtxt);
	if (!muziek) return;

    $('#notation').html ('<div id="leeg" style="height:'+ topSpace +'px">&nbsp;</div>');
    $('#notation').append (muziek);
    $('#notation').append ('<div id="leeg" style="height:'+ topSpace +'px">&nbsp;</div>');
    $('#leeg').click (function () { if (menuVisible) toggleMenu (); else toggleStart (iSeq); });
    //~ $('#notation').css ('overflow-y', opt.hrz ? 'hidden' : 'auto')
    deSvgs = $('#notation svg');        // alle balksystemen (inclusief marges)
    var gs = $('.g');                   // de g met .g class is alleen aanwezig met pagescale != 1
    deSvgGs = gs.length ? gs : deSvgs;
    deSvgs.children ('title').remove ();    // avoid title popup's
    setYSvgs ();
    setScale ();
    deSvgs.click (getNote);
    //~ setVoices (extrStf ? curStaff : 0); // selecteer balk    
    setVoices (curStaff); // selecteer balk, -> setVoices -> mkNtSeq -> markeer (0)
}

function tagSelect (evt) {
    evt.stopPropagation (); // no click bubble to getNote
    var tag = evt.currentTarget;
    if (selectedTag) {
        selectedTag.classList.remove ('tagselect'); // reset selection
        if (tag == selectedTag) { selectedTag = null; return; }
    }
    selectedTag = tag;  // set selection
    tag.classList.add ('tagselect');
}

function plaatsLoper (ix) { // positioneer cursors van alle stemmen (liggende noten!)
    if (opt.lus && (ix < lusStart || ix > lusEnd)) ix = lusStart;   // beperk tot lusgebied
    iSeq = iSeqStart = ix;
    for (i = 0; i <= iSeq; ++i)
        putMarkLoc (ntsSeq [i], 0); // niet aligneren anders teveel scollanimaties
    alignSystem (true); // nu wel aligneren!
}

function drawTags (metMarkeer = 1) {
    function beginMerken () {
        var barTmsSeq = Object.keys (barTimes).sort ((a, b) => a - b).map ((x) => 1 * x);
        barTmsSeq.unshift (0);
        var i = 0, j = 0;
        while (i < barTmsSeq.length && barTmsSeq [i] <= ntsSeq [iSeq].t) i += 1;
        while (j < ntsSeq.length && ntsSeq [j].t < barTmsSeq [i]) j += 1;
        //~ return [barTmsSeq [i - 1], ntsSeq [j - 1].t]    // begin bar ... end bar
        return [ntsSeq [iSeq].t, ntsSeq [j - 1].t]  // selected note ... end bar
    }
    function tijdNootIx (tabc) {
        for (var i = 0; i < ntsSeq.length; ++i) {
            if (ntsSeq [i].t >= tabc) break;
        }
        return i
    }
    function drawtag (tag, i) {
        const wbar = 12;
        var chkmod = $('#chkmod').val ()
        var [isvg, x, y, w, h] = ntsSeq [i].xy;
        var stfys = stfPos [isvg];  // y-coor van onderzijde van alle balken
        var istf = extrStf || chkmod == 4 ? 0 : curStaff;   // istf bovenste balk
        var jstf = istf, lststf = stfys.length - 1;         // jstf onderste balk
        if (chkmod == 3) jstf = Math.min (jstf + 1, lststf); // pianobalk
        if (chkmod == 4) jstf = lststf;                     // alle balken
        var x = tag == atag ? x - wbar : 1*x + w;
        tag.setAttribute ('x', x);
        tag.setAttribute ('y', stfys [istf] - stfHgt [istf]);
        tag.setAttribute ('width', wbar);
        tag.setAttribute ('height', stfys [jstf] - stfys [istf] + stfHgt [istf]);
        tag.style.display = 'initial';  // want was 'none' in $(document).ready
        deSvgGs.eq (isvg).append (tag); // bovenop tekenen, jquery vermijdt dubbele appends
        tag.removeEventListener ('click', tagSelect);
        tag.addEventListener ('click', tagSelect);
    }
    if (ntsSeq.length == 0 || iSeq > ntsSeq.length) return;
    if (opt.lus) { 
        if (opt.atagTijd < 0) [opt.atagTijd, opt.btagTijd] = beginMerken (); // beginselectie
        lusStart = tijdNootIx (opt.atagTijd);
        lusEnd = tijdNootIx (opt.btagTijd);
        drawtag (atag, lusStart);
        drawtag (btag, lusEnd);
        aantalNoten (lusStart, lusEnd + 1);
        if (metMarkeer) markeer (0);    // sets cursor at loopStart (alleen tijdens (ver)plaatsing van lusmarkering)
    } else {
        atag.remove (); btag.remove (); // herhaalde remove doet geen kwaad
        aantalNoten (0, ntsSeq.length - 1); // sets numNotes as side effect, ntsSeq heeft een dummy note aan het eind (zie mergeVoices)
    }
    clearTiming (1);    // -> showScore en reset score bepalende variabelen
}

function berekenTempos () {
    if (times_arr) {    // vorm absolute maattijden in maatduren om
        var tms = times_arr.reduce (function (acc, xs) { return acc.concat (xs.slice (1)); });
        maat_duren = tms.map (function (t, i, ts) { return i == 0 ? 0 : Math.round (100 * (t - ts [i - 1])) / 100; });
    }
    var barTmsSeq = Object.keys (barTimes).sort (function (a, b) {return a - b;}).map (function (a) { return 1 * a; });
    barTmsSeq.unshift (0);
    var i, ix = 0, dabc, tmp;
    for (i = 1; i < maat_duren.length; ++i) {
        dabc = barTmsSeq [i] - barTmsSeq [i - 1];
        tmp = (60 * (dabc / 384) / maat_duren [i]);
        tmp = Math.round (100 * tmp) / 100;
        while (ntsSeq [ix].t < barTmsSeq [i] && ix < ntsSeq.length) {
            ntsSeq [ix].tmp = tmp;
            ix += 1;
        }
    }
    latency += offset_js;
    if (!pre_opt.tempo) // zorg voor correct starttemp, want synctijden gaan voor tempo uit bladmuziek
        pre_opt.tempo = ntsSeq [0].tmp;
}
function mkNtsSeq (iv) {
    function maatStreep (soort) {       // maak tijdreeks voor lijnloper
        barTimes [n.t + offset] = 1;    // maattijden voor metronoom
        if (n.ix in lastBars || soort == 'r' || soort == 'v') // alle sprongen en laatste maatstreep van een regel
            tmsSeq.push ({t: n.t + offset, xy: ntsPos [n.ix], bt: soort});
    }
    ntsSeq = []; barTimes = {}; tmsSeq = []; ixTms = 0;
    var chkmod = $('#chkmod').val ();
    var noten = allNotes.filter (function (n) { return !n.bt && n.v in iv; });  // alle noten die gevolgd worden
    var maten = allNotes.filter (function (n) { return n.bt && n.v in iv; });   // alle maatstrepen
    var times = noten.map (function (n) { return n.t; });
    times.push (maten [maten.length - 1].t + 1);        // extra tijd net voorbij laatste maatstreep
    var tvi = times.shift (), rest = [], prevRest = [], i, j, vce_groups = [], group = [], n, midiNums, chdur, hn;
    for (i = 0; i < allNotes.length; ++i) {             // groepeer noten in tijdgroepen (volgens noten van geselecteerde balk)
        n = allNotes [i];
        if (n.t == tvi) {
            while (n.t == tvi) tvi = times.shift ();    // naar de volgende tijd
            vce_groups.push (group);
            group = []
        }
        if (n.t < tvi) {
            group.push (n);
        }
    }
    vce_groups.push (group);
    var repcnt = 1, offset = 0, repstart = 1, reptime = 0, volta = 0, tvolta = 0, gesprongen = {}, midiNumsKb;
    for (i = 0; i < vce_groups.length; ++i) {
        group = vce_groups [i];
        prevRest = rest;
        rest = [];
        for (j = 0; j < group.length; ++j) {
            n = group [j];
            if (n.bt && n.v == 0) { // als gesprongen moet worden is de rest in prevRest, want maatstreep staat in het begin van de (volgende) groep.
                if (n.bt [0] == ':' && gesprongen [n.t] == 1) { repcnt = 1; gesprongen [n.t] = 2; } // maar 1 keer resetten
                if (n.bt [0] == ':' && !gesprongen [n.t]) {  // sprong terug bij repeat
                    maatStreep ('r');
                    gesprongen [n.t] = 1;   // onthoud score positie van sprong
                    i = repstart - 1; repcnt = 2; offset += n.t - reptime; rest = prevRest;
                    break;
                }
                if (n.bt [n.bt.length - 1] == ':') { repstart = i; reptime = n.t; }
                if (volta && (n.tx || n.bt != '|')) { volta = 0; offset -= n.t - tvolta; }
                if (repcnt == 2 && n.tx == '1') { maatStreep ('v'); volta = 1; tvolta = n.t } // sprong voorwaarts over volta-1 in de tweede herhaling
            };
            if (volta) { rest = prevRest; break; }
            if (n.bt) { maatStreep ('l'); continue; }   // skip barlines
            if (n.v in iv || chkmod == 4) {
                midiNums = {}; midiNumsKb = {};
                n.ns.sort (function (a, b) { return b.nt - a.nt }); // hoogste noot eerst
                if (chkmod == 1) {  // alleen hoogste noot te spelen
                    rest = rest.concat (n.ns.slice (1));
                    hn = n.ns [0];  // hoogste noot
                    midiNums [hn.nt] = hn.nm; midiNumsKb [hn.ntkb] = hn.nm;
                    chdur = hn.dur;
                } else {            // alle noten spelen -> neem max duur van die noten
                    n.ns.forEach (function (m) { midiNums [m.nt] = m.nm;  midiNumsKb [m.ntkb] = m.nm; });
                    chdur = Math.max.apply (Math, n.ns.map (function (m) { return m.dur; }));
                }
                if (Object.keys (midiNums).length > 1) delete midiNums [-1];  // verwijder eventuele rusten uit een akkoord
                ntsSeq.push ({ t: n.t + offset, xy: ntsPos [n.ix], ptc: midiNums, dur: chdur, ptckb: midiNumsKb,
                               rst: prevRest, tmp: n.tmp, vce: n.v, mtr: n.mtr, met: n.met, st: n.st });
                prevRest = rest;    // DUBIEUS!! voor volta midden in gebonden noot (waardoor barline midden in nootgroep met opgebouwde rest-noten)
                                    // mergeVoices gebruikt alleen de prevRest van de eerste noot uit de groep (== gelijktijdige reeks)
            } else {
                n.ns.forEach (function (n) { rest.push ({ t: n.t + offset, nt: n.nt, dur: n.dur }); });
            }
        }
    }
    mergeVoices (rest); // de laatste rest is nog niet toegekend aan een noot in nstSeq
    if (opt.audOn && (times_arr || maat_duren)) berekenTempos ();
    audSeq = [];
    var taud = latency, tnxt, tcur = ntsSeq [0].t, tmpo = ntsSeq [0].tmp;
    for (i = 1; i < ntsSeq.length; ++i) {
        audSeq.push (taud);
        tnxt = ntsSeq [i].t;
        taud += (60 / tmpo) * (tnxt - tcur) / 384;
        tcur = tnxt;
        tmpo = ntsSeq [i].tmp;
    }
    var curNoteTime  = iSeq > 0 && iSeq < ntsSeq.length ? ntsSeq [iSeq].t : 0;
    iSeq = 0;
    for (; iSeq < ntsSeq.length; ++iSeq) {  // zet iSeq zo richt mogelijk bij laatste cursor positie
        if (ntsSeq [iSeq].t >= curNoteTime) break;
    }
    if (iSeq == ntsSeq.length) iSeq -= 1;
    iSeqStart = iSeq;
    markeer (0);
    showScore ();
}

function aantalNoten (i, j) {
    numNotes = 0;
    for (; i < j; ++i) {
        var ns = ntsSeq [i];
        if (-1 in ns.ptc) continue;
        numNotes += Object.keys (ns.ptc).length;
    }
}

function mergeVoices (rest) {
    function addNotesTmsSeq () {
        var seq = [], i = 0, j = 0;
        while (i < ntsSeq.length && j < tmsSeq.length) {
            if (ntsSeq [i].t < tmsSeq [j].t) {
                seq.push ({t: ntsSeq [i].t, xy: ntsSeq [i].xy, bt: ''});
                i += 1;
            } else {
                seq.push (tmsSeq [j]);
                j += 1;
            }
        }
        tmsSeq = seq;
    }
    var seqPrev, seq, i, seqNew = [], ps, ns, v, pskb;
    ntsTimes = {};
    for (i = 0; i < ntsSeq.length; ++i) {
        seq = ntsSeq [i];
        ps = Object.keys (seq.ptc);                 // de midinummers van (de noot | het akkoord)
        pskb = Object.keys (seq.ptckb);             // de midinummers inclusief verbonden noten
        ps = ps.filter (function (mn) { return mn != -1; });            // zonder rusten
        pskb = pskb.filter (function (mn) { return mn != -1; });
        ns = ps.map (function (pt) { return { mn: pt, dur: seq.dur, st: seq.st, nm: seq.ptc [pt] }; }); // midinummers met duur, balk en naam
        if (seqPrev && seqPrev.t == seq.t) {        // merge all notes/rests starting at the same time
            ps.forEach (function (m) {
                if (-1 in seqPrev.ptc) delete seqPrev.ptc [-1];         // remove first rest if any
                seqPrev.ptc [m] = seq.ptc [m];      // verzamel { midinum -> nootnaam }
            });
            pskb.forEach (function (m) {
                if (-1 in seqPrev.ptckb) delete seqPrev.ptckb [-1];     // remove first rest if any
                seqPrev.ptckb [m] = seq.ptckb [m];  // verzamel { midinum -> nootnaam }
            });
            seqPrev.vxy [seq.vce] = seq.xy;
            if (!seqPrev.xy) seqPrev.xy = seq.xy    // onzichtbare rust overschrijven als het kan
            if (seq.xy && 1*seq.xy [1] < 1*seqPrev.xy [1]) seqPrev.xy = seq.xy; // kleinste x-coor van gelijktijdige rusten/noten
            seqPrev.pbk = seqPrev.pbk.concat (ns);                      // verzamel noten voor afspelen
            ns.forEach (n => seqPrev.durs [n.mn] = n.dur);              // midinummer -> duur
            ps.forEach (function (m) { seqPrev.vcs [m] = seq.vce });    // midinummer -> stemnummer
            seqPrev.vms [seq.vce] = ps;                                 // stemnummer -> [midinummers]
            seqPrev.vmskb [seq.vce] = pskb;                             // stemnummer -> [midinummers inclusief verbonden noten]
            seqPrev.nns = Object.keys (seqPrev.vcs).length;             // aantal verschillende noten
        } else {
            seq.pbk = ns;                                               // pbk is alleen voor afspelen
            seq.durs = {};
            ns.forEach (n => seq.durs [n.mn] = n.dur);                  // midinummer -> duur
            seq.vxy = {};
            seq.vxy [seq.vce] = seq.xy;                                 // vierkant per stem
            seq.vcs = {}
            ps.forEach (function (m) {seq.vcs [m] = seq.vce});          // midinummer -> stemnummer
            seq.vms = {}
            seq.vms [seq.vce] = ps;                                     // stemnummer -> [midinummers]
            seq.vmskb = {}
            seq.vmskb [seq.vce] = pskb;                                 // stemnummer -> [midinummers inclusief verbonden noten]
            seq.nns = Object.keys (seq.vcs).length;                     // aantal verschillende noten
            seqPrev = seq;
            seqNew.push (seq);
            ntsTimes [seq.t] = seqNew.length - 1;
        }
    }
    ntsSeq = seqNew;
    var x = ntsSeq [ntsSeq.length - 1]; // dummy noot opdat rest ook gepspeeld wordt
    ntsSeq.push ({ t: x.t + x.dur, xy: x.xy, ptc: x.ptc, dur: x.dur, rst: rest, tmp: x.tmp });
    rMarks.forEach (function (mark) {   // verberg oude markeringen
        mark.setAttribute ('width', 0);
        mark.setAttribute ('height', 0);
    });
    if (keyMark > 0) for (v = 0; v < nVoices; ++v) toetsVeeg (v);   // en verwijder toetsmarkeringen
    addNotesTmsSeq ();
    ntsSeq.forEach ((n, i) => n.iseq = i);  // voor dataopslag in checkOpTijd en checkTiming
}

function setYSvgs () {  // calculate top y-coordinates of all svg's (systems)
    if (deSvgs.length == 0) return;
    var ynot = $('#notation').position ().top - $('#notation').scrollTop ();
    ySvgs = deSvgs.map (function (ix, svg) {
        return $(svg).position ().top - ynot;
    }).get ();  // map returns a jquery object -> get the array
}

function setScale () {
    if (deSvgs.length == 0) return;
    var w_svg, w_vbx, m, scale, svg = deSvgs.get (0);
    var w_svg = svg.getBoundingClientRect ().width;     // width svg element in pixels
    try       { w_vbx = svg.viewBox.baseVal.width; }    // width svg element (vbx coors)
    catch (e) { w_vbx = w_svg; }                        // no viewbox
    if (w_vbx == 0) w_vbx = w_svg;                      // when svg viewport is not scaled to width attribute
    m = deSvgGs.get (0).transform.baseVal;   // scale factor top g-grafic
    scale = m.numberOfItems ? m.getItem (0).matrix.a : 1;   // scale: svg-coors -> vbx-coors
    gScale = ((w_vbx / scale) / w_svg);                 // pixels -> svg-coors
}

function alignSystem (animflag) {   // uitlijnen balken met de rollijn
    if (isvgPrev.length == 0 || isvgPrev [mrkVce] < 0) return;
    var tmargin = $('#rollijn').offset ().top + dottedHeight - $('#notation').offset ().top;
    var istf = extrStf ? 0 : curStaff;
    var y = (stfPos [isvgPrev[mrkVce]][istf] - stfHgt [istf]) / gScale;
    var curSvg = isvgPrev [mrkVce];
    var newTop = Math.round (ySvgs [curSvg] + y - tmargin);
    if (newTop == deNot.scrollTop) return;
    if (opt.hrz > 0) {
        if (hasSmooth) deNot.style ['scroll-behavior'] = 'auto';
        deNot.scrollTop = newTop;
    } else {
        if (hasSmooth) deNot.style ['scroll-behavior'] = animflag ? 'smooth' : 'auto';
        if (hasSmooth || !animflag) deNot.scrollTop = newTop;
        else $(deNot).animate ({ scrollTop: newTop });
    }
}

function putMarkRec (cmd_p) {
    var i, t, cmd = cmd_p.split (',');
    t = parseInt (cmd [0]);
    i = ntsTimes [t];
    if (i >= 0) {
        var tmp = cmd [1]; kwartTempo = tmp;
        var tfac = (60000 / 384) / tmp;
        var nt = ntsSeq [i];
        putMarkLoc (nt);
        if (opt.lncur || opt.hrz) startAbcKlok (nt.t + 0.1, tfac);
        if (gToSynth > 0) {
            var t0 = nu ();
            Object.keys (nt.ptc).forEach (function (midiNum) {
                speel (t0, midiNum, nt.dur * tfac);
            });
            ntsSeq [i + 1].rst.forEach (function (n) {
                var dt1 = (n.t - nt.t) * tfac;
                speel (t0 + dt1, n.nt, n.dur * tfac);
            });
        }
    }
}

function putMarkLoc (n, align = true) {
    var p, isvg, x, y, w, h, mnum, tts, telm, mark, pn, isvgprev, v;
    mrkVce = n.vce;
    for (v in n.vxy) {
        mark = rMarks [v];
        p = n.vxy [v];
        if (!p) {   // n.xy == undefined
            mark.setAttribute ('width', 0);
            mark.setAttribute ('height', 0);
            continue;
        }
        isvg = p[0]; x = p[1]; y = p[2]; w = p[3]; h = p[4];
        isvgprev = isvgPrev [v]
        if (isvg != isvgprev) {
            pn = mark.parentNode;
            if (pn) pn.removeChild (mark);
            deSvgGs.eq (isvg).prepend (mark);
            isvgPrev [v] = isvg;
            if (align) alignSystem (true);
        }
        cursorNotes = 1;
        if (!colorNotes [v]) colorNotes [v] = opt.kleur [v % opt.kleur.length]
        mark.setAttribute ('x', x);
        mark.setAttribute ('y', y);
        mark.setAttribute ('width', w);
        mark.setAttribute ('height', h);
        mark.setAttribute ('fill', colorNotes [v]);
    }
    $tempo.value = Math.round (n.tmp * audSpeed / (n.met / 384));
    if (keyMark > 0) {
        for (v in n.vmskb) toetsVeeg (v);   // eerst alle stemmen vegen
        for (v in n.vmskb) {                // daarna pas tekenen
            n.vmskb [v].forEach (mnum => {
                tts = document.getElementById (tmapI [mnum]);
                telm = document.getElementById (txtmapI [mnum]);
                prevKeys [v].push ([tts, telm, mnum]);
                tts.style.fill = opt.kleur [v % opt.kleur.length];
                var [step, acc] = n.ptckb [mnum]; // n.ptc == { midinum -> [stap, verhoging] }
                telm.innerHTML = stepnm [step] + accnm [acc + 2];   // nootnaam
            });
        }
    }
}

function putMarkSlv (n, tmp) {   // we komen hier alleen als isMaster == 1
    if (n.t != slaveTime) sok.send (n.t + ',' + tmp);    // alleen naar slaven
    slaveTime = n.t;        // vermijd dubbele positioneringen
}

function nu () {
    if (gToSynth == 1) return audioCtx.currentTime * 1000;   // millisecs !
    else return new Date().getTime() - performance.timing.navigationStart;  // current time w.r.t. browser start
}

function zend (midiMsg, tijd) {
    if (gToSynth == 0) return;              // no sound
    if (gToSynth == 1 && gGolvenKlaar) {    // internal synthesizer
        var mtype = midiMsg [0] & 0xf0,
            vol = midiMsg [2],
            midiNum = midiMsg [1], src;
        tijd /= 1000;   // millisec -> sec
        if (mtype == 0x80) op (midiNum, tijd);
        if (mtype == 0x90) {
            if (vol > 0) src = neer (midiNum, vol, tijd);
            else op (midiNum, tijd);
        }
        if (mtype == 0xB0 && midiNum == 64) {
            if (vol >= 64) {
                sustain = 1;
            } else {
                sustain = 0;
                Object.keys (sustained).forEach (function (mnum) {
                    op (mnum, tijd);
                });
                sustained = {};
            }
        }
    }
    if (gToSynth == 2)                      // external synthesizer (MIDI output)
        if (gOutput) gOutput.send (midiMsg, tijd);
}

function speel (tijd, noot, dur) {  // tijd en duur in millisecs
    if (noot == -1) return; // een rust
    var midiMsg = [0x90, noot, gSpeelVol];
    zend (midiMsg, tijd);
    midiMsg [2] = 0;
    zend (midiMsg, tijd + dur);
}

function simPlay () {
    if (timeNotes < 0) {                // start knop niet gebruikt
        toggleStart (-1);
        if (!opt.nocnt) return;
    }
    var nt = checkNotes;
    var tfac = (60000 / 384) / kwartTempo;
    var tijd = nu ();
    if (!nt.pbk) return;    // end of music ?
    nt.pbk.forEach (function (n) {
        if (n.mn == -1) return; // een rust
        var midiMsg = [0x90, n.mn, gFollowVol];
        echoMIDIMessage ({ data: midiMsg });    // checks en als opt.echo -> zend ()
        midiMsg [2] = 0;
        zend (midiMsg, tijd + n.dur * tfac - 1);    // MIDI off message
    });
    for (var m in nt.vcs) {
        if (m == -1) continue; // geen rusten en al goede noten
        var v = nt.vcs [m];
        colorNotes [v] = opt.ntGoed;
        rMarks [v].setAttribute ('fill', opt.ntGoed);
    }
}

function clearTiming (withData, inlus) {
    if (withData) {
        timingData = [];
        errorCount = goodNotes = lateNotes = korteNoten = 0
        showScore ();
        for (var k in pMarks) pMarks [k].remove (); // verwijder permanente markeringen
        pMarks = {}; pMarksType = {};
        veegFoutTypes ();
    }
    if (!inlus) timeNotes = -1; // in doorgaande lus mode is timeNotes geldig en moet niet gereset worden
}

function showStats (dfr) {
    var sx1 = timingData.map (function (x, i) { return [i, x.laat]; });
    var sx2 = timingData.map (function (x, i) { return [i+0.2, x.duur [0]]; });
    var sx3 = timingData.map (function (x, i) { return [i+0.4, x.duur [1] || null]; });
    var sx4 = timingData.map (function (x, i) { return [i+0.6, x.duur [2] || null]; });
    perfElm.style.display = 'block';
    var options = { yaxes: [
        { position: "left", font: {color: 'rgba(0,128,0,1)'}, min: -500, max: 500 },
        { position: "right", font: {color: '#602010'}, min: -100, max: 100 }
    ]};
    $.plot ('#flot', [
        { label: 'timing', color: 'rgba(0,128,0,0.5)', data: sx1, bars: {show: true, barWidth: 0.5, align: 'left'}, yaxis:1 },
        { label: 'duration', color: '#a0522d', data: sx2, bars: {show: true, barWidth: 0.1, align: 'left'}, yaxis:2 },
        { label: '', color: '#a0522d', data: sx3, bars: {show: true, barWidth: 0.1, align: 'left'}, yaxis:2 },
        { label: '', color: '#a0522d', data: sx4, bars: {show: true, barWidth: 0.1, align: 'left'}, yaxis:2 }
    ], options);
}

function startAbcKlok (tAbc, tfac, dopos) {
    var isvgPrev = -1, stop = 0, tAbcNxt, pn;
    function rolLinks (x) {
        x = x / gScale - deNot.scrollLeft;
        var dx = x - deNot.clientWidth / 3;
        deNot.scrollLeft += dx;
    }
    function tekenSvgLoper (isvgPrev, isvg, x, y, h) {
        if (!opt.lncur && !opt.hrz) {
            pn = loper.parentNode;
            if (pn) pn.removeChild (loper);
            cancelAnimationFrame (timerAbc);
            return; 
        }
        if (isvg != isvgPrev) {
            pn = loper.parentNode;
            if (pn) pn.removeChild (loper);
            deSvgGs.eq (isvg).prepend (loper);
        }
        var ys = stfPos [isvg]
        y = ys [0] - stfHgt [0];
        h = ys [ys.length - 1] - y;
        var kleur = opt.oneclr ? opt.ntWacht : colorNotes [0];
        loper.setAttribute ('x', x);
        loper.setAttribute ('y', y-6);
        loper.setAttribute ('width', 2);
        loper.setAttribute ('height', h+12);
        if (kleur) loper.setAttribute ('fill', kleur);  // soms undefined ???
        loper.setAttribute ('fill-opacity', opt.lncur ? opt.mat : 0);
        if (opt.hide2cur && opt.nocur) loper.setAttribute ('fill-opacity', 0);
    }
    function tekenLoper (tabc, tabcnxt) {   // huidige abctijd en van 25 msec verder
        var xy1, xy2, isvg, tms, tmsprv, xfac, x, xleft, xright, y, h;
        tms = tmsSeq [ixTms];
        if (tabc < tms.t) { ixTms = 0; tms = tmsSeq [0]; }
        while (tabc >= tms.t) {     // tms -> begin volgende noot/rust
            ixTms += 1;
            tms = tmsSeq [ixTms];
        }
        if (tabcnxt >= tms.t) {     // overslaan bij dopos == 1 (tabcnxt == 0)
            stop = 1;               // stoppen in abcTik anders gaan we voorbij begin volgende noot
            if (tms.t > ntsSeq [iSeq].t) {                      // sprong terug bij einde lus
                x = startAbcKlok (ntsSeq [iSeq].t + 0.1, 0, 1); // alleen lijn tekenen en rollen
                return x;
            }
            if (tmsSeq [ixTms + 1] && tmsSeq [ixTms + 1].xy [1] < 1 * tms.xy [1]) { // sprong terug bij herhaling
                x = startAbcKlok (tms.t + 0.1, 0, 1);           // alleen lijn tekenen en rollen
                return x;
            }
        }
        tmsprv = tmsSeq [ixTms-1];
        xfac = (tabc - tmsprv.t) / (tms.t - tmsprv.t)
        x = tmsprv.xy;
        isvg = x[0]; y = x[2]; h = x[4];
        xleft =  1 * x[1];
        xright = 1 * tms.xy [1];
        x = xleft + (xright - xleft) * xfac;
        tekenSvgLoper (isvgPrev, isvg, x, y, h);
        isvgPrev = isvg;
        if (opt.hrz > 0) rolLinks (x);
    }
    function abcTik (stempel) {
        if (stempelOud == 0) stempelOud = stempel;
        tAbc += (stempel - stempelOud) / tfac;  // tijdverchil > 60 Hz = 17 msec
        tAbcNxt = tAbc + 25 / tfac; // 25 msec verder, kijk of eind van de nootduur
        tekenLoper (tAbc, tAbcNxt); // zijeffect: stop -> 1
        stempelOud = stempel;
        if (stop && opt.volgmod != 6) cancelAnimationFrame (timerAbc);  // midi => volgende startAbcKlok
        else timerAbc = requestAnimationFrame (abcTik);
    }
    cancelAnimationFrame (timerAbc);
    var stempelOud = 0;
    if (dopos) tekenLoper (tAbc);
    else timerAbc = requestAnimationFrame (abcTik);
}

function markeer (inc, play, start) {
    iSeq += inc;
    if (iSeq < 0) iSeq = 0;     // inc kan -1 zijn
    curNote = iSeq > 0 && !start ? ntsSeq [iSeq - 1] : ntsSeq [iSeq];   // de noot die nu klinkt
    var nt = ntsSeq [iSeq];             // de volgende noot

    var t0 = nu ();
    var tmp = kwartTempo;
    nuKwartTempo = kwartTempo;
    var playDt = t0 - prevTime;     // tijdspanne vorige noot - huidige noot
    var f = playDt / prevDt;        // de timing fout (kleiner dan 1 is sneller)
    if (inc && !prevPlayRest        // niet tijdens positioneren of in het begin of na rust of tijdens count-in
    && adaptiveMode && f < fHoog && f > fLaag) {    // verwerp grove fouten
        if (f >= 1) f = 1 + (f - 1) * 0.5;
        else f = 1 - (1 - f) * 0.5;
        tmp = Math.round (tmp / f);
        tmp = minTempo > tmp ? minTempo : tmp;
        kwartTempo = tmp;
        audSpeed = tmp / curNote.tmp;   // snelheidsverandering
        audElm.playbackRate = audSpeed;
    }
    var tfac = (60000 / 384) / tmp;
    nuTfac = tfac;
    var tfac_nxt = tfac;
    var t1 = curNote.t;                 // abc tijd van laatst gespeelde noot
    var dt = (nt.t - t1) * tfac;        // delta abc tijd * tempo = delta echte tijd in msec
    if (curNote.tmp != nt.tmp) {        // tempo uit bladmuziek verandert
        kwartTempo = nt.tmp * audSpeed;  // tempo voor volgende noot
        tfac_nxt = (60000 / 384) / kwartTempo;
    }
    var dt_nxt = nt.dur * tfac_nxt;     // duur van de volgende noot

    var tfacCor = 1, dtUncor = dt;
    if (inc && hasAud && !audElm.ended) {
        if (audElm.paused) {    // wanneer "play one note before start" is geselecteerd 
            speelAudio (-1);    // en na afloop van timer4 (speler gestopt), of gewoon bij start
        }
        var dtAud = audElm.currentTime - audSeq [iSeq - 1]; // tijd van de huidige noot! iSeq wijst naar volgende noot.
        if (opt_url.lat) $('#deb').html ((dtAud * 1000).toFixed (0) + ' msec') // + ', ' + (playDt - prevDt).toFixed (0));
        if (!adaptiveMode) {
            dt -= dtAud * 1000; // aligneer audio en score time
            if (dt < 0) dt = 0; // we kunnen niet terug in de tijd
            tfacCor = dt / dtUncor; // correctie voor de lijnloper
            if (tfacCor < 0.3) tfacCor = 0.3;
        }
        clearTimeout (timer4);  // reset de foutwachter na iedere noot
        timer4 = setTimeout (function () { audElm.pause (); }, dt + opt.tmo); // pauseer audio als speler stopt
    }

    var dtlast = -1;
    if ((inc || play) && !opt.rstmute) nt.rst.forEach (function (n) {     // rest noten tot aan volgende te spelen noot
        var dt1 = (n.t - t1) * tfac;
        speel (t0 + dt1, n.nt, n.dur * tfac);
        if (!isMaster || dt1 == dtlast) return;         // zend niet meer dan 1 sync request per tijdstip
        dtlast = dt1;
        setTimeout (function () { putMarkSlv (n, tmp) }, dt1);   // send sync requests for all n.t times
    });

    if (iSeq == ntsSeq.length - 1) {    // dummy noot voor de laatste nt.rst die hierboven gespeeld is
        timer6 = setTimeout (() => { 
            for (var vce in errorMarks) markeerPermanent (checkNotes, vce, errorMarks [vce]); // en teken de laatste markeringen
            toggleStart (-1); 
        }, dt); // stop meteen
        if (opt.lncur || opt.hrz) startAbcKlok (t1, tfac * tfacCor);   // maar laat de lijnloper doorgaan
        return
    }
    if (opt.lus && (iSeq > lusEnd || iSeq < lusStart)) {
        if (opt.lusstop && inc) {
            setTimeout (() => { toggleStart (-1); }, dt); // stop na sprong
            return;
        } else if (lusStart < ntsSeq.length) {  // lusStart kan groter zijn na balkwissel
            iSeq = iSeqStart = lusStart;    // ga door na sprong
            nt = ntsSeq [iSeq]; // de volgende noot
            setTimeout (() => { // na uitklinken van laatste lusnoot
                clearTiming (1, 1);
                if (hasAud && loopt) {  // herstart de audio bij lusStart
                    //~ audElm.pause ();
                    speelAudio ();
                }
            }, dt);
        }
    }

    prevPlayRest = inc == 1 && (-1 in nt.ptc); // speel rust tijdens volgen
    if (isPlaying || prevPlayRest) {    // speel ook de te volgen noten
        if (!opt.mute) nt.pbk.forEach (function (noot) {
            speel (t0 + dt, noot.mn, noot.dur * tfac_nxt);
        });
        clearTimeout (timer1);
        timer1 = setTimeout (function () { markeer (1); }, dt);
    } else if (metTimeout && inc) {
        var dtmin = (nt.t - t1) * tfacmin;
        clearTimeout (timer1);
        timer1 = setTimeout (function () { markeer (1); }, dtmin);
    }
    var twin = dt > dt_nxt ? dt_nxt * 0.5 : dt * 0.5;
    if (opt.exact && exactEnabled) twin = 0;
    var dt2 = dt - twin;    // dt2 == dt, dt3 == 0
    var dt3 = metDelay ? dt - dt2 : 0;
    if (inc == 0) {     // positioneren
        gChk = {};
        errorMarks = {}
        checkNotes = nt;            // deze noten moeten gechecked worden
        colorNotes = {};
        if (!gCountInPb)    // niet schoonmaken aan het einde van de count down
            geelNeer = {};  // goed gespeelde noten die voor het checkwindow zijn ingedukt (en nog steeds neer)
        plaatsLoper (iSeq);
        if (hasAud) audElm.currentTime = audSeq [iSeq]; // also set audio player position
        curNote = nt;   // -> the right tempo in toggleStart
        if (opt.lncur || opt.hrz) startAbcKlok (nt.t + 0.1, tfac, 1);
        if (isMaster) { putMarkSlv (nt, tmp); slaveTime = -1; }
    } else {
        if (opt.lncur || opt.hrz) startAbcKlok (t1, tfac * tfacCor);   // start ABC klok op de huidige ABC tijd
        timer2 = setTimeout (function () {
            if (isPlaying) {                    // alleen met volgmod == 6
                for (var m in checkNotes.vcs) { // check voor overgeslagen noten
                    if (m in gChk || m == -1) continue;     // geen rusten en al goede noten
                    errorMarks [checkNotes.vcs [m]] = 1;    // verzamel de stemmen
                }
            }                                   // when cursor moves, place the permanent error marks
            for (var vce in errorMarks) markeerPermanent (checkNotes, vce, errorMarks [vce]);
            errorMarks = {};
            gChk = {};
            checkNotes = nt;        // deze noten moeten gechecked worden
            colorNotes = {};
            cursorNotes = 0;        // wacht tot de cursor voor deze noot getekend is
            timeNotes = t0 + dt;    // the time in msec when nt will sound
            clearTimeout (timer3);  // lange noot gevolgd door korte, als korte te vroeg wordt gespeeld
            timer3 = setTimeout (function () {  // alleen voor de master
                putMarkLoc (nt);
                //~ console.log ('mrk: ' + nt.pbk[0].mn + ' ' + (nu () - timeNotes).toFixed (0));
                for (var mn in geelNeer) checkMidi (mn, 1);  // want te vroeg gespeeld en dus niet gecheckt
                geelNeer = {};
            }, dt3);                // cursor verschijnt (na dt2 + dt3)
        }, dt2);                    // check window begint na helft duur klinkende noot
        if (isMaster) timer5 = setTimeout (function () { putMarkSlv (nt, tmp); }, dt);   // naar de slaven
    }
    prevTime = t0;
    prevDt = dt;
    if (metroOn && (t1 in barTimes || t1 == 0)) metroHigh (1);   // synchroniseer de metronoom
}

function clearPbTimers () {
    clearTimeout (timer1); clearTimeout (timer2);
    clearTimeout (timer3); clearTimeout (timer5); clearTimeout (timer6);
    cancelAnimationFrame (timerAbc);
}

function playBack () {  // alleen met volgmod == 6 (playback, zie toggleMetro, gCountInPb)
    isPlaying = 1;      // equivalent met volgmod == 6
    if (hasAud) speelAudio ();
    markeer (0, 0, 1);
}

function speelAudio (di = 0) {
    audElm.currentTime = audSeq [iSeq + di];;
    audElm.play ();
}

function toggleStart (stoppos, withCountIn = true) {
    function audioReady (evt) {
        audElm.removeEventListener ('canplaythrough', audioReady);
        logerr ('sufficient audio data loaded');
        audElm.pause ();
        audElm.currentTime = audSeq [0];
        waitDlg ('#unlkDlg', 0);
        hasAud = 1;
        toggleMetro (withCountIn);
    }
    
    
    if (!ntsSeq.length) return;
    if (startKnop.value == 'stop' || stoppos >= 0) {
        startKnop.value = 'start';
        if (MidiOutput.length > 0){
            download(MidiOutput, inFileName + "_output.txt", 'txt');
            MidiOutput = [];
            MidiOutput.push(["Start!",0].toString() + "\n");
        }
        loopt = 0;
        stopMetro ();       // also stops audio
        isPlaying = 0;
        clearPbTimers ();   // when in playback mode
        timeNotes = -1;     // metronome start at next note
        iSeq = stoppos >= 0 ? stoppos : iSeqStart;
        iSeqStart = iSeq;   // zet ook de permanente startpositie
        markeer (0, 0, 1);  // meteen op de noot beginnen
        if (opt.ftyp) toonFoutTypes ();
        if (stoppos < 0) {  // when at end of preload, or when explicitly stopping
            window.parent.postMessage ($error.innerHTML, '*');  // when in iframe
        }
    } else {                // => toggleMetro => metroHigh => loopt = 1
        startKnop.value = 'stop';
        kwartTempo = curNote.tmp * audSpeed;
        audElm.playbackRate = audSpeed;
        nuKwartTempo = kwartTempo;
        if (audioCtx.state != 'running') {
            audioCtx.resume ().then (() => {
                logerr ('audio context now running');
            });
        }
        if (hasAudUrl && audElm.readyState < 4) {
            timeNotes = 0;  // vermijd herhaalde middle-C-start in checkMidi
            audElm.addEventListener ('canplaythrough', audioReady);
            waitDlg ('#unlkDlg', 1);
            speelAudio (); // -> canplaythrough event -> pause()
        } else {
            if (hasAudUrl) hasAud = 1;
            toggleMetro (withCountIn);
            startTime = nu();
        }
    }
}

function keyDown (e) {
    var key = e.key;
    if (key == 'Escape') {
        if (perfElm.style.display == 'block') perfElm.style.display = 'none';
        else toggleMenu ();
    }
    if (menuVisible && key != 'm') return;
    switch (key) {
    case 'ArrowLeft': case 'Left': markeer (-1); break;
    case 'ArrowUp': case 'Up': changeStaff (-1); break;
    case 'ArrowRight': case 'Right': simPlay (); break;
    case ' ': e.preventDefault (); simPlay (); break;
    case 'ArrowDown': case 'Down': changeStaff (1); break;
    case 'c': toggleStart (-1); break;
    case 'm': $('#mbar').click (); break;
    }
}

function initScore () {
    function rgb (k) {
        if (k[0] != '#') return k;
        var r = parseInt (k[1]+k[1], 16);
        var g = parseInt (k[2]+k[2], 16);
        var b = parseInt (k[3]+k[3], 16);
        return `rgba(${r},${g},${b},${opt.mat})`
    }
    kleuren = {};
    opt.ntKleur.forEach ((k, i) => { kleuren [i+1] = k; });
    elmSgd = document.getElementById ('sgd');   // score elements
    elmSe1 = document.getElementById ('se1');
    elmSe2 = document.getElementById ('se2');
    elmSe3 = document.getElementById ('se3');
    elmPct = document.getElementById ('pct');
    elmStot = document.getElementById ('stot');
    elmSe1.style.background = rgb (kleuren [1]);    // fout
    elmSe2.style.background = rgb (kleuren [2]);    // te laat of te vroeg
    elmSe3.style.background = rgb (kleuren [4]);    // te kort of te lang
    showScore ();
}

function showScore () {
    if (!numNotes) return;  // er zijn nog geen noten
    if (!elmSgd) return;    // laden document is nog niet klaar
    var perr = errorCount * opt.pnlerr, ptim = lateNotes * opt.pnltim, pkrt = korteNoten * opt.pnlkrt;
    var score = 100 * (goodNotes - perr - (opt.tmng ? ptim : 0) - (opt.tmng2 ? pkrt : 0)) / numNotes;
    if (score < 0) score = 0;
    var [k1,k2,k3] = ['rgba(255,0,0,0.3)','rgba(0,191,255,0.5)','rgba(0,255,0,0.5)'];
    var x = score.toFixed (0);
    if (opt.scal) {
        elmSgd.innerHTML = goodNotes;
        elmSe1.innerHTML = perr;
        elmSe2.innerHTML = ptim;
        elmSe3.innerHTML = pkrt;
        elmStot.innerHTML = numNotes;
    }
    elmPct.innerHTML = x;
    elmPct.style.background = score < 50 ? k1 : score < 85 ? k2 : k3;
    if (opt.lus) localStorage.setItem ('loopscore', x);
    else localStorage.setItem ('score', x);
}

function checkTiming (mnum) {
    var df = neerTijd [mnum] - timeNotes; // verschil gespeeld - muziekaal in msec
    var dfa = Math.abs (df);    // absoluut
    if (opt.tmng && dfa > opt.late1) {
        lateNotes += 1;
        if (dfa > opt.late2) lateNotes += 1;
        var v = checkNotes.vcs [mnum];
        errorMarks [v] = (errorMarks [v] || 0) | 2;
    }
    timingData [checkNotes.iseq] = { laat: df, duur: [] };
}

function checkMidi (mnum, laat) {
    var n = checkNotes, v;
    if (!n) return; // == undefined -> markeer niet aangeroepen -> dolayout niet klaar
    if (!laat) neerTijd [mnum] = nu (); // onthou de aanslagtijd
    var nts = Object.keys (n.ptc);      // alle te spelen noten
    if  (timeNotes < 0 &&               // start knop niet gebruikt
        (mnum == 60 || mnum in n.ptc)){ // middle C or correct note
            if (gCountIn == 1) {
                geelNeer [mnum] = 1;    // noot te vroeg gespeeld tijdens neertellen
                return;                 // wacht op einde neertellen
            }
            //toggleStart (-1, mnum == 60);
            //if (!opt.nocnt && mnum == 60) return;   // because count-in starts
            if (-1 in n.ptc) return;    // starten op een rust
    }
    if (!loopt) return;                 // alleen controleren als muziek loopt
    if (mnum == opt.stopat) {           // optional midi key to stop
        toggleStart (-1);
        return;
    }
    if (Object.keys (gChk).length == n.nns || -1 in n.ptc) {
        geelNeer [mnum] = 1;    // noot te vroeg gespeeld (of tijdens een rust)
        return                  // wordt straks in markeer gechecked
    }
    if (mnum in n.vcs) {
        gChk [mnum] = 1; // gespeelde noot was goed
        neerNt [mnum] = n;  // onthou voor duur bij optillen
        v = n.vcs [mnum];
        if (n.vms [v].every (n => n in gChk)) { // alle noten van deze stem zijn goed
            colorNotes [v] = opt.ntGoed;
            if (cursorNotes) rMarks [v].setAttribute ('fill', opt.ntGoed);
            checkTiming (mnum); // only check once per voice
        }
    } else {
        errorCount += 1;
        showScore ();
        for (var m in n.vcs) {
            if (m in gChk || m == -1) continue; // geen rusten en al goede noten
            v = n.vcs [m];
            colorNotes [v] = opt.ntFout;        // de rest wordt rood
            errorMarks [v] = (errorMarks [v] || 0) | 1;
            if (cursorNotes) rMarks [v].setAttribute ('fill', opt.ntFout);
        }
    }
    if (Object.keys (gChk).length == n.nns) { // alle noten goed
        goodNotes += n.pbk.length;
        showScore ();
        if (!isPlaying) markeer (1);    // vermijd dubbele markeer(1) tijdens playback
    }
}

function checkOpTijd (mnum) {
    if (!opt.tmng2) return;
    if (!neerNt [mnum]) return;
    var n = neerNt [mnum];
    var duur = n.durs [mnum] * nuTfac;  // nootduur in millisecondes
    var d = neerTijd [mnum] + duur - nu ();
    var fout = 100 * d / duur;
    delete neerNt [mnum];
    var fouta = fout < 0 ? -fout : fout;
    var da = d < 0 ? -d : d;
    if (fouta > opt.kort && da > opt.minkort) {
        korteNoten += 1;
        showScore ();
        var v = n.vcs [mnum];
        if (n.iseq == checkNotes.iseq) errorMarks [v] = (errorMarks [v] || 0) | 4;
        else {  // veel te laat opgetild, de cursor is al verder
            var k = n.iseq + ',' + v, m = pMarks [k];   // is er al een markering?
            if (m) {
                pMarksType [k] |= 4;            // voeg 4 toe aan de kleur index
                m.setAttribute ('fill', kleuren [pMarksType [k]]);
            } else markeerPermanent (n, v, 4);  // nieuwe markering
        }
    }
    var td = timingData [n.iseq];
    if (td) td.duur.push (fout); // de data hoort bij deze iseq in ntsSeq
}

function markeerPermanent (n, v, kleur) {   // nootrecord, stemnummer, kleur index
    if (-1 in n.ptc) { console.log ('rust fout', n.iseq, v, kleur); return; }
    var k = n.iseq + ',' + v;
    var p = n.vxy [v];
    if (p) {
        var isvg = p[0], x = p[1], y = p[2], w = p[3], h = p[4];
        var mark = document.createElementNS ('http://www.w3.org/2000/svg','rect');
        mark.setAttribute ('x', x);
        mark.setAttribute ('y', y);
        mark.setAttribute ('width', w);
        mark.setAttribute ('height', h);
        mark.setAttribute ('fill', kleuren [kleur]);
        mark.setAttribute ('fill-opacity', opt.metmrk ? opt.mat : 0);
        deSvgGs.eq (isvg).prepend (mark);
        pMarks [k] = mark; pMarksType [k] = kleur;  // onthoud het fouttype
    }
}

function toonFoutTypes () {
    for (var k in pMarks) {
        var t = pMarksType [k];
        var m = pMarks [k];
        var [i, v] = k.split (',');
        var [isvg, x, y, w, h] = ntsSeq [i].vxy [v].map (x => 1 * x);
        var txt = document.createElementNS ('http://www.w3.org/2000/svg','text');
        txt.setAttribute ('x', x + w);
        txt.setAttribute ('y', y + h);
        txt.setAttribute ('font-size', '12');
        txt.setAttribute ('class', 'tfout');
        var n = document.createTextNode (t);
        txt.appendChild (n);
        m.parentElement.append (txt);
    }
}

function veegFoutTypes () {
    var xs = document.getElementsByClassName ('tfout');
    Array.from (xs).forEach (e => e.remove ());
}

function testPort () {
    var t = nu ();
    for (var i = 0; i < 3; ++i) {   // play a test C5,D5,E5
        speel (t, 0x3C + 2*i, 300);
        t += 310; // milli secs
    }
}

function download(data, filename, type) {
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob) // IE10+
        window.navigator.msSaveOrOpenBlob(file, filename);
    else { // Others
        var a = document.createElement("a"),
                url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);  
        }, 0); 
    }
}

function echoMIDIMessage (event) {
    var mtype = event.data [0], midiNum = event.data [1], vol = event.data [2];
    if (mtype == 0xfe) return;
    mtype &= 0xf0;      // discard channel
    if (mtype == 0x80) { delete geelNeer [midiNum]; checkOpTijd (midiNum); }
    if (mtype == 0x90) {
        if (vol > 0) checkMidi (midiNum);
        else { delete geelNeer [midiNum]; checkOpTijd (midiNum); }
    }
    if (opt.echo) zend (event.data, nu ());
    //MidiOutput.push(event.data.concat(nu() - startTime).toString() + "\n")
    var outlist = [nu() - startTime, midiNum, vol]
    MidiOutput.push(outlist.toString() + "\n")
    console.log(MidiOutput)
    //fs.writeFile('/test_out.txt', [event.data,nu()], { flag: 'a+' }, err => {});
}

function onMIDISuccess (midiAccess) {
    logerr ('MIDI ready! Listening to following input ports:');
    gMidi = midiAccess;
    var ins = midiAccess.inputs, ix = 0;
    ins.forEach (function (port) {
        port.onmidimessage = echoMIDIMessage;   //  listen to all input ports
        logerr (ix + ': ' + port.name + ', "' + port.manufacturer + '"');
        ix++;
    });
    logerr ('The following output ports are present:');
    var portsel = $('#portsel');
    var outs = midiAccess.outputs, ix = 0;
    outs.forEach (function (port) {
        logerr (ix + ': ' + port.name + ', "' + port.manufacturer + '"');
        portsel.append ('<option value="' + ix + '">' + port.name + '</option>');
        gOutPorts.push (port);
        gOutput = port; // select the last port
        ix++;
    });
}

function onMIDIFailure (msg) {
    logerr ('Failed to get MIDI access - ' + msg);
}

function onPortSelect (menuAction) {
    var p = $('#portsel').val();
    if (p == 'synth') {
        gToSynth = 1;
        if (menuAction) laadNootHulp (midiUsed); // alleen na gebruikersinteractie
    } else if (p == 'nosound') {
        gToSynth = 0;
    } else {
        gToSynth = 2;
        var ix = parseInt (p);
        gOutput = gOutPorts [ix];
    }
    testPort ();
}

function tryMidi () {
    if ('requestMIDIAccess' in navigator)
        navigator.requestMIDIAccess ().then (onMIDISuccess, onMIDIFailure);
    else {
        var m = 'Web MIDI API is not supported in this browser: try Chrome or Opera';
        m += ' (on iOS you can try the '
        m += '<a href="https://apps.apple.com/us/app/web-midi-browser/id953846217">'
        m += 'Web Midi Browser</a>)'
        $('#notation').append ('<br>' + m);
        logerr (m);
    }
    $('#ble').prop ('disabled', !('bluetooth' in navigator));
    if (!parsePreload ()) setUrlOptions ();
}

function setVoices (istf) {
    if (!(istf in gStaves)) istf = 0;
    clearPbTimers ();
    var voices = {};                // verzamel stemmen van de huidige balk
    gStaves [istf].forEach (function (iv) { voices [iv] = 1; });
    if ($('#chkmod').val () == 3) { // select both staves of grand staff
        var jstf = istf;
        if (istf + 1 in gStaves) jstf = istf + 1;
        else if (istf - 1 in gStaves) { jstf = istf - 1; istf = jstf; }
        gStaves [jstf].forEach (function (iv) { voices [iv] = 1; });
    }
    isvgPrev.forEach (function (x,i,arr) { arr [i] = -1; });    // verwijder vorige indices
    if (gAbcTxt) mkNtsSeq (voices); // maak een tijdreeks van de verzamelde stemmen
    curStaff = istf;
    drawTags (0);    // -> aantalNoten, cursors worden niet verplaatst (metMarkeer == 0)
}

function changeStaff (up) {
    curStaff += up;
    if (curStaff < 0) curStaff = 0;
    if (curStaff >= gStaves.length) curStaff = 0;
    setVoices (curStaff);
}

function toggleMenu () {
    menuVisible = 0;
    $('#menu').toggle (false)
    $('.down').toggleClass ('down', false); 
}

function showMenu (n) {
    if (menuVisible == n) { toggleMenu (); return; }  // klik op de knop van zichtbaar menu
    toggleMenu ();   // menuVisible => 0
    menuVisible = n;
    $('#menu').toggle (true)
    $('#hfd div').toggle (false);
    $('#hfd div:nth-child('+n+')').toggle (true);
    $('#btns > div > div:nth-child('+n+')').toggleClass ('down', true);
    $('#btns > #mbar').toggleClass ('down', true);  // voor de oude html file
}

var metroTempo, metroTimer = 0, metroOn;
function metroLow () {
    $mtrpad.style.fill = 'none';
    metroIsLow = 1;
    if (metroOn || gCountIn) {
        metroTimer = setTimeout (metroHigh, metroTempo);
    }
}

function stopMetro () {
    gCountIn = 0; gCountInPb = metroOn = false;
    clearTimeout (metroTimer);
    $('#countin').toggle (false);
    metroLow ();
    if (hasAud) audElm.pause ();
}

function toggleMetro (withCountIn) {
    var cntTab = {'2/2':2, '3/2':3, '4/2':4, '2/4':2, '3/4':3, '4/4':4, '5/4':5, '6/8':2, '7/8':3, '9/8':3, '11/8':4, '12/8':4, '3/8': 4}
    $('#start').blur ();        // want anders start spatietoets opnieuw
    if (metroOn || gCountIn) {  // stop metronoom
        stopMetro ();
        return;
    }
    clearTiming (1);    // also clear statistics
    var k = curNote.mtr.top + '/' + curNote.mtr.bot;
    gCountIn = withCountIn && !opt.nocnt ? cntTab [k] + 1 : 1;
    gCountInPb = opt.volgmod == 6;
    $('#countin').html ('<b>' + gCountIn + '</b>').toggle (true);
    metroHigh ();
}

function metroHigh (luid) {
    $mtrpad.style.fill = 'red';
    //~ if (metroIsLow) metroNeer (luid);   // vermijd dubbele klik door synchronisatie (eind van markeer)
    metroNeer (luid);
    metroIsLow = 0;
    metroTempo = (30000 / nuKwartTempo) * (curNote.met / 384);
    clearTimeout (metroTimer);
    metroTimer = setTimeout (metroLow, metroTempo);
    if (gCountIn > 0) {
        gCountIn -= 1;
        $('#countin').html ('<b>' + gCountIn + '</b>');
        if (gCountIn == 0) {
            $('#countin').toggle (false);
            loopt = 1;
            metroOn = opt.metro;    // ga alleen door als menu item aangevinkt
            prevTime = timeNotes = nu ();   // volg meteen de geselecteerde noot
            if (gCountInPb) {
                playBack ();        // gCountInPb => geelNeer wordt niet schoongemaakt
                gCountInPb = 0;     // one shot
            } else if (opt.eenvoor) {
                markeer (0, 1);     // speel eerst de rest-noten voor de geselecteerde noot.
            } else {
                prevDt = 0;
                if (-1 in ntsSeq [iSeq].ptc) markeer (1);   // starting on a rest
            }
            for (var mn in geelNeer) checkMidi (mn, 1); // te vroeg gespeeld
            geelNeer = {};
        }
    }
}

function decodeer (jsondata, klaar) {
    var byteString = atob (jsondata);
    var ab = new ArrayBuffer (byteString.length);
    var ia = new Uint8Array (ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    audioCtx.decodeAudioData (ab, function (buffer) {
        klaar (buffer);
    }, function (error) {
        console.log ('decode error: ' + error);
        gGolvenKlaar = 0;
    });
}

function schakelAudio (aan) {
    opt.audOn = aan;
    $('#spkron').toggle (aan);  // wissel plaatje
    $('#spkrof').toggle (!aan);
    audElm.muted = !aan;
    if (gAbcSave) setVoices (curStaff); // herbereken de tempos in ntsSeq
}

function check_preload (mod) {
    if (mod.pre_opt) {
        media_file = mod.media_file;
        pre_opt = mod.pre_opt;
        abc_arr = mod.abc_arr;
        for (var k in mod.defaults) {
            if (!(k in pre_opt)) pre_opt [k] = mod.defaults [k];
        }
    }
    initOptions ();
    readAbcOrXML (abc_arr.join ('\n'));
    waitDlg ('#wait', 0, !sok || opt_url.mstr);
}

function parsePreload () {
    var parstr, xmlfnm = '', preload = '', r, p, ps, i, m = '', host, elm;
    parstr = window.location.href.replace ('?dl=0','').split ('?'); // look for parameters in the url;
    if (r = parstr [0].match (/:\/\/([^/:]+)/)) host = r [1];
    if (parstr.length > 1) {    // preload media and score
        ps = parstr [1].split ('&');
        for (i = 0; i < ps.length; i++) {
            p = ps [i].replace (/d:(\w{15}\/[^.]+\.)/, 'https://dl.dropboxusercontent.com/s/$1');
            if (r = p.match (/tmp=([\d.]*)/)) opt_url.tempo = parseFloat (r [1]);
            else if (r = p.match (/stf=(\d+)/)) opt_url.staff = parseInt (r [1]);
            else if (r = p.match (/mod=(\d+)/)) opt_url.chkmod = parseInt (r [1]);
            else if (r = p.match (/fmd=(\d+)/)) opt_url.volgmod = parseInt (r [1]);
            else if (r = p.match (/opa=(\d)/)) opt_url.kbopa = parseInt (r [1]);
            else if (r = p.match (/line=(\d+)/)) opt_url.line = parseInt (r [1]);
            else if (r = p.match (/pw=(\d+)/)) opt_url.pw = parseInt (r [1]);
            else if (r = p.match (/scl=([\d.]*)/)) opt_url.scl = parseFloat (r [1]);
            else if (r = p.match (/hrz=(\d+)/)) opt_url.hrz = parseInt (r [1]);
            else if (r = p.match (/tb=([\d.]*)/)) opt_url.btime = parseFloat (r [1]);
            else if (r = p.match (/te=([\d.]*)/)) opt_url.etime = parseFloat (r [1]);
            else if (r = p.match (/begin=([\d]+)/)) opt_url.begin = parseInt (r [1]);            
            else if (r = p.match (/shrnk=([\d.]*)/)) opt_url.shrnk = parseFloat (r [1]);
            else if (r = p.match (/lat=([+-]?[\d]+)/)) opt_url.lat = parseInt (r [1]);
            else if (r = p.match (/vol=([\d]*)/)) opt_url.vol = parseInt (r [1]);
            else if (r = p.match (/pnlerr=([\d.]*)/)) opt_url.pnlerr = parseFloat (r [1]);
            else if (r = p.match (/pnltim=([\d.]*)/)) opt_url.pnltim = parseFloat (r [1]);
            else if (r = p.match (/pnlkrt=([\d.]*)/)) opt_url.pnlkrt = parseFloat (r [1]);
            else if (r = p.match (/late1=([\d]*)/)) opt_url.late1 = parseInt (r [1]);
            else if (r = p.match (/late2=([\d]*)/)) opt_url.late2 = parseInt (r [1]);
            else if (r = p.match (/kort=([\d]*)/)) opt_url.kort = parseInt (r [1]);
            else if (r = p.match (/minkort=([\d]*)/)) opt_url.minkort = parseInt (r [1]);
            else if (r = p.match (/stopat=([\d]*)/)) opt_url.stopat = parseInt (r [1]);
            else if (r = p.match (/hinfo=(\d+)/)) opt_url.hinfo = parseInt (r [1]);
            else if (r = p.match (/mat=([\d.]+)/)) opt_url.mat = parseFloat (r [1]);
            else if (r = p.match (/kleur=([#\dabcdef,]+)/)) opt_url.kleur = r [1].split (',');
            else if (r = p.match (/tmo=(\d+)/)) opt_url.tmo = parseInt (r [1]);
            else if (p == 'ip' && host) opt_url.ipadr = host;
            else if (p == 'ipm' && host) { opt_url.ipadr = host; opt_url.mstr = 1; }
            else if (p == 'metro') opt_url.metro = 1;
            else if (p == 'nocnt') opt_url.nocnt = 1;
            else if (p == 'nodel') opt_url.delay = 0;
            else if (p == 'keyb') opt_url.keys = 1;
            else if (p == 'ksh') opt_url.mark = 1;
            else if (p == 'nosnd') opt_url.portsel = 'nosound';
            else if (p == 'nobar') opt_url.nobar = 1;
            else if (p == 'nomenu') opt_url.nomenu = 1;
            else if (p == 'extr') opt_url.extract = 2;
            else if (p == 'extrg') opt_url.extract = 3;
            else if (p == 'nocur') opt_url.nocur = 1;
            else if (p == 'mute') opt_url.mute = 1;
            else if (p == 'rstmute') opt_url.rstmute = 1;
            else if (p == 'nosm') hasSmooth = false;
            else if (p == 'ctrls') opt_url.ctrls = 1;
            else if (p == 'tmng') opt_url.tmng = 1;
            else if (p == 'tmng2') opt_url.tmng2 = 1;
            else if (p == 'lncur') opt_url.lncur = 1;
            else if (p == 'oneclr') opt_url.oneclr = 1;
            else if (p == 'nolyr') opt_url.nolyr = 1;
            else if (p == 'hide2cur') opt_url.hide2cur = 1;
            else if (p == 'metmrk') opt_url.metmrk = 1;
            else if (p == 'ftyp') opt_url.ftyp = 1;
            else if (p == 'scal') opt_url.scal = 1;
            else if (p == 'noecho') opt_url.echo = 0;
            else preload = p;
            if (/(\.xml$)|(\.abc$)|(\.mxl$)/.test (preload)) { xmlfnm = preload; preload = ''; }
            if (/(\.mp3$)|(\.mp4$)|(\.ogg$)|(\.webm$)/.test (preload)) { m = preload; preload = ''; }
        }
        if (m) {
            media_file = decodeURIComponent (m).replace ('www.dropbox', 'dl.dropboxusercontent');
        }
        if (preload || xmlfnm) { waitDlg ('#wait', 1); }
        if (xmlfnm) {           // force loading xml as plain text
            $.get (xmlfnm, '', null, 'text').done (function (data, status) {
                logerr ('preload: ' + status);
                readAbcOrXML (data);
                waitDlg ('#wait', 0);
            }).fail (function (jqxhr, settings, exception) {    // same origin policy
                $('#wait').append ('\npreload failed: ' + settings);
            });
        } else if (preload) {   // get the javasript preload file
            getPreload (preload);
        }
    }
    return preload || xmlfnm;
}

function getPreload (preload) {
    if (preload.indexOf ('dropbox.com') >= 0) preload += '?dl=1';
    if (preload.indexOf ('data:text') != 0 && preload.indexOf ('http') == -1) preload = './' + preload;
    import (preload).then (function (mod) {
        logerr ('preload loaded');
        check_preload (mod);
    }).catch (function (err) {
        $('#wait,#err').append ('\npreload failed:\n' + err);
    });
}

function setMenuItem (id, waarde) {
    var item = document.getElementById (id);
    var type = item ? item.type : undefined;
    if (waarde != undefined) opt [id] = waarde;
    if (type == 'checkbox') item.checked = opt [id];
    if (type == 'number' || type == 'select-one') item.value = opt [id];
    if (waarde != undefined) doOption (id);
}

function setUrlOptions () {
    for (var id in pre_opt) opt [id] = pre_opt [id];
    for (    id in opt_url) opt [id] = opt_url [id]; // url parameters gaan voor
    for (    id in optMenu) setMenuItem (id);
    for (    id in opt) doOption (id, 0);
    hasAud = 0;
    hasAudUrl = 0;
    latency = opt.lat / 1000;
    addMediaFile ();
    volElm.addEventListener ('change', function () {
        volElm.blur ();
        opt.vol = volElm.value;
        masterVol = opt.vol / 25;
    });
    volElm.value = opt.vol;
    masterVol = opt.vol / 25;   // dus max vol == 4
    laadNootHulp (midiUsed);
    if (gAbcTxt) {  // gAbcSave set in dolayout -> gAbcTxt set in doModel, 
        if (opt.pw || opt.hrz || opt.nolyr) dolayout (gAbcSave, iSeq);   // iSeq == 0 || opt.begin
        else doLayout (gAbcTxt, iSeq);
        doOption ('begin', 1);  // doLayout -> mkNtsSeq zet curNoteTime verkeerd (verkeerde ntsSeq als staff != 0)
    }
    if (opt_url.ipadr) { webSokOpen (opt_url.ipadr); if (!opt_url.mstr) deSvgs.off (); }    // verwijder event handlers in een slaaf
    if (demoDlg) {
        $('#demoDlg span').html (demoDlg);
        $('#demoDlg').toggle (true)
        $('#demoDlg button').on ('mouseup touchend keydown blur', function (evt) {
            evt.stopPropagation (); // don't trigger keyDown
            $('#demoDlg').toggle (false)
            $('#demoDlg button').off ('mouseup touchend keydown blur');
        }).focus ();
    }
    initScore ();
}

function addMediaFile () {
    if (media_file) {
        play_start = opt.btime >= 0 ? opt.btime : 0;
        play_end = opt.etime > 0 ? opt.etime : 0;
        audElm.src = media_file;
        if (opt.ctrls) audElm.controls = true;
        hasAudUrl = 1;
        $('#spkr').toggle (true);
    }
    mediaFnm = media_file || '';
}

function savePreload () {
    if (!gAbcSave) return;
    var fnm = 'media_file = "' + (mediaFnm || '') + '";\n';
    var of = '', ds = '', dss = [];
    if (maat_duren) {
        of = 'offset_js = ' + offset_js.toFixed (2) + ';\n';
        while (maat_duren.length) dss.push (maat_duren.splice (0, 10).toString ());
        ds = 'maat_duren = [' + dss.join (',\n') + '];\n';
    }
    var abcSave = gAbcSave.split ('\n');
    var abcpln = abcSave.map (function f (x) { return JSON.stringify (x); });
    abcpln = 'abc_arr = [' + abcpln.join (',\n') + '];\n';
    opt.staff = curStaff + 1;
    opt.line = Math.round (100 * $('#rollijn').position ().top / $('body').height ());
    opt.begin = iSeqStart;
    var xs = ('pre_opt = ' + JSON.stringify (opt) + ';').split (',');
    var os = [], regel = '';
    while (xs.length) {
        while (regel.length < 80 && xs.length) regel += xs.splice (0, 1) + ',';
        os.push (regel); regel = '';
    }
    os = os.join ('\n').replace (/,$/,'\n'); // laatste komma wordt nieuwe regel (einde statement)
    var res = fnm + of + os + ds + abcpln;
    var dataUrl = 'data:text/plain;charset=utf-8;base64,' + btoa (unescape (encodeURIComponent (res)));
    $('#saveDlg pre').text (res);
    $('#saveDlg').toggle (true);
    $('#div3 a').attr ('href', dataUrl);
    $('#saveok').focus ();
}

function setVolgMod (menuAction) {
    metTimeout = false; minTempo = 0; $('#mtplab').toggle (false);
    adaptiveMode = 1;
    setFHoog ();    // want kan door case 7 veranderd zijn
    opt.volgmod = $('#volgmod').val ();
    switch (opt.volgmod) {
    case '5':
        $('#mtplab').toggle (true);
        minTempo = opt.mtpo;
        break;
    case '6': adaptiveMode = 0; break;
    case '7': adaptiveMode = 0; break;
    }
    if (menuAction) setMenuItem ('delay', !adaptiveMode);
    clearTiming ();  // clear de meter
}

function setFHoog () {
    fHoog = metTimeout ? 10 : 3;
    fLaag = 1 / fHoog;
}

function setAudSpeed () {
    audSpeed = curNote ? kwartTempo / curNote.tmp : 1;
    audElm.playbackRate = audSpeed;
}

function toggleKeyboard () {
    keyboardVisible = $('#keys').prop ('checked');
    $('#marklbl').toggle (keyboardVisible);
    //~ $('#kbopalbl').toggle (keyboardVisible);
    $('#keyb').toggle (keyboardVisible);
    $('#zoom').toggle (keyboardVisible);
    $('#kblft').toggle (keyboardVisible);
    $('#kbrgt').toggle (keyboardVisible);
    $('#rek').toggle (keyboardVisible);
    if (keyboardVisible) verbreedToetsen (0);   // geef alle octaaf-svgs een precieze percentuele breedte
    else centerKeyboard (); // else -> verbreedToetsen doet al centerKeyboard
    resizeNotation ();
}

function ttsNeer (evt) {
    evt.stopPropagation ();
    evt.preventDefault ();  // no mouse events when touchDev
    evt.target.style.fill = 'red';
    var i = tmap [evt.target.id];
    toetsNeer [i] = 1;
    var midiMsg = [0x90, i, gFollowVol];
    echoMIDIMessage ({ data: midiMsg });
}

function ttsOp (evt) {
    evt.stopPropagation ();
    evt.preventDefault ();  // no mouse events when touchDev
    var i = tmap [evt.target.id];
    if (i in toetsNeer) {
        delete toetsNeer [i];
        var midiMsg = [0x90, i, 0];
        echoMIDIMessage ({ data: midiMsg });
        evt.target.style.fill = keyColor (i);
    }
}

function keyColor (i) { // midi number
    return (i % 12) in blackKeys ? 'black' : (i == 60 ? 'white' : 'white');
}

function verbindToetsen () {
    blackKeys = { 1:1, 3:1, 6:1, 8:1, 10:1 }  // midinum % 12 van zwarte toetsen
    var stepmap = [0,2,4,5,7,9,11,1,3,6,8,10], midmap = [];
    for (var oct = 0; oct < 8; ++oct) { // bereken de afbeelding van toets-id naar midi nummer
        for (var i = 0; i < 12; ++i) {
            midmap.push (oct * 12 + stepmap [i] + 12);
        }
    }
    var ks = $('#toetsen rect');
    for (var i = 0; i < ks.length; ++i) {
        var k = $ (ks [i])
        tmap [ks[i].id] = midmap [i];
        tmapI [midmap [i]] = ks[i].id;
        k.on ('mousedown touchstart',  ttsNeer);
        k.on ('mouseup touchend',  ttsOp);
        k.on ('mouseleave',  ttsOp);
        ks [i].style.fill = keyColor (midmap [i]);
    }
    var ks = $('#toetsen text');
    for (var i = 0; i < ks.length; ++i) {
        txtmapI [midmap [i]] = ks[i].id;    // midinum -> tekst ID
    }
}

function centerKeyboard () {
    var w = $('#toetsen')[0].scrollWidth - $('body').width ();
    $('#keyb').scrollLeft (w / 2);
}

function resizeNotation () {
    var eh = keyboardVisible && keyTransparency == 0 ? $('#keyb').height () : $('#err').height ();
    var bh = $('body').height (), ih = $('#info').height ();
    ih = Math.round (100 * ih / bh);
    eh = Math.round (100 * eh / bh);
    $('#notation').css ('height', (100 - ih - eh) + '%');
}

function verbreedToetsen (n) {
    var ts = $('#toetsen svg')
    var w = ts.width ();
    w = 100 * w / $('body').width ();
    w += n;
    ts.css ('width', w + '%');
    centerKeyboard ();
}

function maakToetsen () {
    var oct = $('#octaaf');
    for (var i = 0; i < 8; ++i) {
        var oc = oct.clone ();
        oc.find ('rect').each (function (ix, rx) {
            rx.id += '-' + i;   // geef alle rects in de kloon een unieke id (voor verbindToetsen)
        });
        oc.find ('text').each (function (ix, rx) {
            rx.id += '-' + i;   // geef alle tekst in de kloon een unieke id (voor verbindToetsen)
        });
        $('#toetsen').append (oc);
    }
    verbindToetsen ();          // geef iedere toets een event handler
}

function zoomToetsen (evt) {
    if (!keyboardVisible) return;
    if (evt.target.id != 'zoom') return;
    evt.stopPropagation (); evt.preventDefault ();
    var touchDev = evt.type == 'touchstart';
    var doel = $('#zoom');
    var x1 = touchDev ? evt.originalEvent.touches[0].clientX : evt.pageX;
    var y1 = touchDev ? evt.originalEvent.touches[0].clientY : evt.pageY;
    var bh = $('#keyb').height ();
    var sl = $('#keyb').scrollLeft (), xscroll = 0;
    doel.css ('cursor', 'row-resize').toggleClass ('spel', true);
    doel.on ('mousemove touchmove', function (evt) {
        evt.stopPropagation (); evt.preventDefault ();
        var x2 = touchDev ? evt.originalEvent.touches[0].clientX : evt.clientX;
        var y2 = touchDev ? evt.originalEvent.touches[0].clientY : evt.clientY;
        if (Math.abs (y2 - y1) > Math.abs (x2 - x1)) {
            var y = bh + y1 - y2;
            var h = $('body').height ();
            var p = Math.floor (100 * y / h);    // height percentage
            $('#keyb').css ('height', p + '%');
            $('#zoom').css ('bottom', p + '%');
            $('#kblft').css ('bottom', p + '%');
            $('#kbrgt').css ('bottom', p + '%');
        } else {
            $('#keyb').scrollLeft (sl + x1 - x2);
            xscroll = 1;
        }
    });
    doel.on ('mouseup touchend mouseleave', function (evt) {
        evt.stopPropagation (); evt.preventDefault ();
        doel.css ('cursor', 'initial').toggleClass ('spel', false);
        doel.off ('mousemove touchmove mouseup touchend mouseleave');
        resizeNotation ();
    });
}

function zoomInfo (evt) {
    if (evt.target.id != 'info' && evt.target.id != 'error' && evt.target.id != '') return;
    evt.stopPropagation (); evt.preventDefault ();
    var touchDev = evt.type == 'touchstart';
    var doel = $('#info');
    var y1 = touchDev ? evt.originalEvent.touches[0].clientY : evt.pageY;
    var bh = doel.height ();
    var nh = $('#notation').height ();
    doel.css ('cursor', 'row-resize').toggleClass ('spel', true);
    doel.on ('mousemove touchmove', function (evt) {
        evt.stopPropagation (); evt.preventDefault ();
        var h = $('body').height ();
        var y2 = touchDev ? evt.originalEvent.touches[0].clientY : evt.clientY;
        var y = bh + y2 - y1;
        var p = Math.floor (100 * y / h);    // height percentage
        doel.css ('height', p + '%');
        opt.hinfo = y;  // bewaar nieuwe hoogte in pixels
        resizeNotation ();
    });
    doel.on ('mouseup touchend mouseleave', function (evt) {
        evt.stopPropagation (); evt.preventDefault ();
        doel.css ('cursor', 'initial').toggleClass ('spel', false);
        doel.off ('mousemove touchmove mouseup touchend mouseleave');
        resizeNotation ();
        alignSystem ();
    });
}

function toetsNamen (mark) {
    var accs = ['&#9837;&#9837;','&#9837;','','&#9839;','&#9839;&#9839;'];
    var t = '<tspan dy="-5">x</tspan>';
    accnm = accs.map (a => t.replace ('x', a));
    var ts = document.querySelectorAll ('#toetsen text');
    for (var i = 0; i < ts.length; ++i) {
        ts [i].setAttribute ('dx', mark == 2 ? -3 : 0);
    }
    if (keyMark == 1)       { stepnm = 'C,D,E,F,G,A,B'.split (','); }
    else if (keyMark == 2)  { stepnm = 'Do,Re,Mi,Fa,So,La,Ti'.split (','); }
}

function toetsVeeg (v) {
    if (!prevKeys [v]) return;
    prevKeys [v].forEach (([tts, telm, mnum]) => {
        tts.style.fill = keyColor (mnum);
        telm.innerHTML = '';
    });
    prevKeys [v] = [];
}

function verbindSchuifknop (id) {
    var links = 0, isNeer = 0;
    function maakWijder () {
        if (isNeer) {
            verbreedToetsen (links ? -1 : 1)
            setTimeout (maakWijder, 200);
        }
    }
    function neer (evt) {   // evt.currentTarget is de svg (kblft of kbrgt)
        evt.stopPropagation ();
        evt.preventDefault ();
        evt.currentTarget.children [1].style.fill = 'red';  // cirkel is kind[1]
        links = evt.currentTarget.id == 'kblft';
        isNeer = 1;
        maakWijder ();
    }
    function op (evt) {
        evt.stopPropagation ();
        evt.preventDefault ();
        if (!isNeer) return
        evt.currentTarget.children [1].style.fill = 'rgba(0,255,0,0.3)';  // de beginkleur, zie def in html
        isNeer = 0;
    }
    var k = $(id);
    k.on ('mousedown touchstart',  neer);
    k.on ('mouseup touchend mouseleave',  op);
}

function lijn_shift (evt) {
    evt.preventDefault();
    var touchDev = evt.type == 'touchstart';
    var doel = $('#rollijn');
    doel.css ('cursor', 'row-resize').toggleClass ('spel', true);
    doel.on ('mousemove touchmove', function (evt) {
        var h = $('#notation').offset ().top;
        var y = touchDev ? evt.originalEvent.touches[0].clientY : evt.clientY;
        $('#rollijn').css ('top', y - dottedHeight / 2 + 'px');
        alignSystem ();
    });
    doel.on ('mouseup touchend mouseleave', function (evt) {
        doel.off ('mousemove touchmove mouseup touchend mouseleave');
        doel.css ('cursor', 'initial').toggleClass ('spel', false);
    });
}

function logerr (s, nnl) {
    if (!nnl) s += '\n';
    $('#err').append (s);
}
function webSokOpen (ip) {
    if (sok) { logerr ('websocket already open');; return; }
    var url = 'ws://' + ip + ':' + 8091 + '/';
    sok = new WebSocket (url);
    sok.onmessage = function (event) {
        var d = event.data;
        //~ logerr (d);
        if (d == 'master') {
            $('#mbar').css ('background', 'rgba(255,0,0,0.2)');
            isMaster = 1;
        } else if (!isMaster) putMarkRec (d);   // alleen in slaven ontvangen
    }
    sok.onerror   = function (event) { logerr ('socket error (server inaccessible?)');  sok = null; }
    sok.onopen    = function (event) { 
        $('#mbar').css ('background', 'rgba(0,255,0,0.2)');
        if (opt_url.mstr) sok.send ('master');
        logerr ('connection opened');
    }
    sok.onclose   = function (event) {
        $('#mbar').css ('background', '');
        logerr ('connection closed: ' + event.code); sok = null;
        isMaster = 0;
    }
}

function initOptions () {
    opt = {};
    gAbcTxt = '';
    for (var id in optMenu) opt [id] = optMenu [id]; // kopieer default waarden 
    for (    id in optRest) opt [id] = optRest [id];
    for (    id in optMenu) setMenuItem (id);
    for (    id in opt) doOption (id, 0);
}

function doOption (id, layout, menuAction) {
    switch (id) {
    case 'tempo': kwartTempo = opt.tempo; setAudSpeed (); break;
    case 'mtpo': minTempo = opt.mtpo; break;
    case 'chkmod': if (gAbcSave && layout) setVoices (curStaff); break;
    case 'delay': metDelay = opt [id]; break;
    case 'exact':
        exactEnabled = opt.volgmod != 6
        var e = document.getElementById ('exact');
        if (e) e.disabled = !exactEnabled;
        var e = document.getElementById ('delay');
        if (e) e.disabled = opt.exact && exactEnabled; // alleen als element bestaat
        break;
    case 'volgmod': setVolgMod (menuAction); doOption ('exact'); break;
    case 'portsel': onPortSelect (menuAction); break;
    case 'keys': toggleKeyboard (); break;  // -> sets keyboardVisible
    case 'mark':
        keyMark = opt [id];
        toetsNamen (keyMark);
        for (var v in prevKeys) toetsVeeg (v);          // veeg alle toetsmarkeringen uit
        if (ntsSeq.length) putMarkLoc (ntsSeq [iSeq]);  // markeer evt. de geselecteerde noot
        break;
    case 'kbopa':
        keyTransparency = opt [id] / 10;
        $('#keyb').css ('opacity', 1 - keyTransparency);
        resizeNotation ();
        break;
    case 'hinfo':
        var h = $('body').height ();
        var p = Math.floor (100 * opt.hinfo / h);   // pixels -> rounded percentage
        $('#info').css ('height', p + '%');
        resizeNotation ();
        break;
    case 'pw':
        opt.pw = parseFloat (opt [id]);
        if (opt.pw < 5) { delete opt.pw; $('#pw').val (''); }
        if (gAbcSave && layout) dolayout (gAbcSave);
        break;
    case 'scl':
        opt.scl = parseFloat (opt [id]);
        if (opt.scl < 0.1 || opt.scl > 2) { delete opt.scl; $('#scl').val (''); }
        if (gAbcSave && layout) dolayout (gAbcSave);
        break;
    case 'nolyr': if (gAbcSave && layout) dolayout (gAbcSave, iSeq); break;
    case 'lus': if (layout) drawTags (); $('#stoplbl').toggle (!!opt.lus); break;
    case 'metro': if (metroOn) metroOn = false; break;  // stop metronoom
    case 'extract':
        extrStf = opt.extract - 1;
        if (extrStf == 0) scoreOrig = '';
        if (gAbcSave && layout) doLayout (gAbcTxt);
        break;
    case 'micuse': if (opt.micuse) micOn (); else micOff (); break;
    case 'audOn': schakelAudio (opt.audOn);
    case 'gain': setVol (); break;
    case 'drempel': setDrempel (); break;
    case 'minlev': setMinLev ();
    case 'bass': setBass (); break;
    // ---------- de opties zonder menu equivalent ------------
    case 'staff': curStaff = opt.staff - 1; if (gAbcSave && layout) setVoices (curStaff); break;
    case 'line': $('#rollijn').css ('top', opt.line + '%'); alignSystem (); break;
    case 'begin': iSeq = iSeqStart = opt.begin; if (gAbcSave && layout) markeer (0); break;
    case 'lncur':
        if (gAbcSave && opt.lncur) startAbcKlok (ntsSeq [iSeq].t + 0.1, 0, 1);
        if (!opt.lncur) loper.setAttribute ('fill-opacity', 0);
        break;
    case 'nocur':
        rMarks.forEach (mrk => mrk.setAttribute ('fill-opacity', opt.nocur ? 0 : opt.mat));
        if (opt.hide2cur) loper.setAttribute ('fill-opacity', opt.nocur ? 0 : (opt.lncur ? opt.mat : 0));
        break;
    case 'metmrk':   // toon/verberg permanente markeringen
        for (var k in pMarks) pMarks [k].setAttribute ('fill-opacity', opt.metmrk ? opt.mat : 0);
        break;
    case 'drpuse': dropuse (); break;
    case 'nobar': $('#info').toggle (false); $('#err').toggle (false); $('#notation').css ('height', '100%'); alignSystem (); break;
    case 'nomenu': $('#btns').toggle (false); break;
    case 'hrz':
        if (layout) opt.hrz = opt.hrz ? 50 : 0; // via menuklik aangeroepen
        $('#pwlbl').toggle (opt.hrz == 0)
        $('#scllbl').toggle (opt.hrz > 0)
        if (gAbcSave && layout) {   // layout -> via menuklik aangeroepen
            dolayout (gAbcSave);
        }
        break;
    case 'tmng': case 'tmng2': case 'scal':  // calculation layout changes
        $('#sgd').parent ().toggle (!!opt.scal);
        $('#se2').parent ().toggle (!!(opt.tmng && opt.scal));
        $('#se3').parent ().toggle (!!(opt.tmng2 && opt.scal));
        $('#stot').parent ().toggle (!!opt.scal);
        showScore ();
        break;
    case 'ftyp': opt.ftyp ? toonFoutTypes () : veegFoutTypes ();
        break;
    }
}

function checkMenu (evt) {
    var type = $(this).attr ('type') || this.type;
    var id = $(this).attr ('id');
    if (type == 'checkbox') opt [id] = $(this).prop ('checked');  // update the option object
    if (type == 'number' || type == 'select-one' || type == 'range') opt [id] = $(this).val ();
    doOption (id, 1, 1);    // id, do layout, is menu action
}

function dropuse () {
    function grey (b) { $('#drpuse').prop ('checked',!b); $('#drpuse').attr ('disabled',b); $('#drplbl').css ('color',b?'#aaa':'#000'); }
    function chgknop () {
        var du = $('#drpuse').prop ('checked');
        $('.dropbox-dropin-btn').css ('display', du ? 'inline-block' : 'none');
        $('#fknp').css ('display', du ? 'none' : 'inline-block');
    }
    if (typeof (Dropbox) == 'undefined') {
        grey (true);
        var u = 'https://www.dropbox.com/static/api/2/dropins.js', dknp;
        $.ajax ({url: u, dataType: 'script', cache: true}).done (function () {
            grey (false);
            Dropbox.init ({appKey: 'ckknarypgq10318'});
            dknp = Dropbox.createChooseButton ({
                success: readDbxFile, extensions: ['.xml', '.mxl', '.abc', '.js'],
                cancel: function() {}, linkType: "preview", multiselect: false
            });
            $('#flbl').append (dknp);
            chgknop ();
        });
    } else chgknop ();
}

function ASMDF (frame) {          // Average Squared Mean Difference Function
	var frame_len = frame.length, i, lag, d, diff, diff_sum = 0, lag_best = -1, left, lag_half;
	var lag_max = Math.floor (frame_len / 2), p, y1, y2, y3, max_lev = 0; // lag_max = fftSize / 4 == graaf_breedte;
	var diff_min = Infinity;
	diffs = new Array (lag_max);

    lag_half = lag_max / 2;
	for (lag = 3; lag < lag_max; lag++) {
		diff = 0;
        left = Math.floor (lag_half - lag / 2); // symmetrisch rond midden frame
        if (frame [left] > max_lev) max_lev = frame [left];
		for (i = left; i < left + lag_max; i++) {
			d = frame [i] - frame [i + lag];
            diff += d * d;      // som van kwadratische verschillen
		}
        diff_sum += diff;       // cumulatieve som van de diffs tot zover
        diff *= (lag - 2) / diff_sum; // een soort normering, begint met diff == diff_sum en lag == 3 -> diff = 1
		diffs [lag] = diff;
        if (diff < diff_min) {
            diff_min = diff;
            lag_best = lag;
        } else if (diff_min < 0.1) break;
	}
    // parabool door drie samples rond lag_best (zie pitch_interpolate.wxm)
    y3 = diffs [lag_best + 1]; y2 = diffs [lag_best]; y1 = diffs [lag_best - 1];
    p = (y1 - y3) / (y3 - 2 * y2 + y1) / 2 || 0;    // y3 of y1 kunnen undefined zijn
    return { period: lag_best + p, diff: diff_min, max: max_lev };
}

function getPitch (prev_period) {
	analyser.getFloatTimeDomainData (frame);
    var p = ASMDF (frame);
    var dp = p.period / prev_period;
    var eqp = dp < 1.06 && dp > 0.94;   // periode binnen halve nootstap gelijk aan vorige
    var midiMsg, freq;
 	if (p.diff < drempel && p.diff > 0 && p.max > minLevel || p.diff < 0.65 && eqp) { // p.diff > 0 want frame samples == 0 in de rewind van de audio
        freq = audioCtx.sampleRate / p.period;
        if (!eqp && !micNote) {
            micNote = Math.round (69 + 12 * Math.log (freq / 440) / Math.log (2));
            midiMsg = [0x90, micNote, gFollowVol];
            echoMIDIMessage ({ data: midiMsg });
            console.log ('micNote on:', micNote, p.period.toFixed (2), p.diff.toFixed (2));
        }
        prev_period = p.period;
    } else {
        if (micNote) {
            midiMsg = [0x90, micNote, 0];   // volume 0 == MIDI off
            echoMIDIMessage ({ data: midiMsg });
            console.log ('micNote off', micNote, eqp, p.diff.toFixed (2));
            micNote = 0;
        }
        prev_period = 1;
    }
    tmrID = setTimeout (function () { getPitch (prev_period); }, 0);
}

function setMinLev () {
    var db = parseInt (opt.minlev);
    var w = (40 + db) * 2.5;
    var maxw = $('#mxlvl').width ();
    $('#levmrk').css ('width', maxw * w / 100 + 'px');
    minLevel = Math.pow (10, db / 20);
    $('#mval').html (db);
}

function setDrempel () {
    var v = parseFloat (opt.drempel);
    drempel = v / 10;
    $('#dval').html (v);
    blauweLijn ();
}

function setVol () {
    var v = parseFloat (opt.gain);
    if (gainKnoop) gainKnoop.gain.value = Math.pow (10, (v + 1) / 2 - 3);
    $('#gval').html (v);
}

function setBass () {
    graaf_breedte = $('#bass').prop ('checked') ? 1024 : 512;
    if (analyser) setFFTsize ();
}

function setFFTsize () {
    analyser.fftSize = 4 * graaf_breedte;
    var frame_len = analyser.frequencyBinCount;
    frame = new Float32Array (frame_len);
}

function micOff () {
    if (deStroom) {
        deStroom.getAudioTracks ().forEach (function (trk) { trk.stop (); });
        bronKnoop.disconnect ();
        deStroom = null;
        cancelAnimationFrame (rafID);
        clearTimeout (tmrID);
    }
    $('#micuse').prop ('checked', false);
    $('#micKnop').css ('background', ''); $('#micon').css ('color', '#666');
}

function micOn () {
    function doMicStream (stroom) {
        deStroom = stroom;  // onthouden want de stroom moet gestopt worden!
        bronKnoop = audioCtx.createMediaStreamSource (stroom);
        gainKnoop = audioCtx.createGain ();
        analyser = audioCtx.createAnalyser ();
        setFFTsize ();
        bronKnoop.connect (gainKnoop);
        gainKnoop.connect (analyser);
        setVol ();
        kiesLus ();
        $('#micKnop').css ('background', 'orange'); $('#micon').css ('color', '#c33');
        setMenuItem ('delay', false);   // zet de delay uit
    }
    function fout (err) { alert ('Can not access the microphone!\n' + err.name + ": " + err.message); micOff (); }
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia ({audio: true}).then (doMicStream).catch (fout);
    } else {
        navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
        try { navigator.getUserMedia ({audio: true}, doMicStream, fout); }
        catch (err) { alert ("getUserMedia() not supported by your browser -> no microphone\n" + err.name + ": " + err.message); micOff (); }
    }
}

function toggleMic () {
    var id = 'micuse';
    var e = document.getElementById (id);
    opt [id] = e.checked = !e.checked;
    doOption (id, 0);
}

function kiesLus () {
    if (!deStroom) return;
    cancelAnimationFrame (rafID);
    clearTimeout (tmrID);
    if (micDlgVisible) {
        rafID = requestAnimationFrame (function () { updatePitch (1); });
    } else {
        tmrID = setTimeout (function () { getPitch (1); }, 0);
    }
}

function showMicDlg (b) {
    $('#tuner').toggle (b);
    if (b) { $('#micuse').focus (); setMinLev (); } // de eerste keer staat het streepje niet goed
    if (menuVisible) toggleMenu ();
    micDlgVisible = b;
    kiesLus ();
}

function updatePitch (prev_period) {
    var t0 = performance.now ();
	analyser.getFloatTimeDomainData (frame);
	var p = ASMDF (frame);
    var dp = p.period / prev_period;
    var eqp = dp < 1.06 && dp > 0.94;

    var freq = audioCtx.sampleRate / p.period;
    var note_num = 69 + 12 * Math.log (freq / 440) / Math.log (2);
    var dt = (performance.now () - t0).toFixed (2);
 	if (p.diff < drempel && p.diff > 0 && p.max > minLevel || p.diff < 0.65 && eqp) {   // p.diff > 0 want frame samples == 0 in de rewind van de audio
        prev_period = p.period;
        var note = Math.round (note_num);
        var note_letter = noteStrings [note % 12] + Math.floor (note / 12);
        nootElem.innerHTML = note_letter;
        var cent = Math.round ((note_num - note) * 100);
        centArr.push (cent);
        centAvg += (cent - centArr.shift ()) / centLen;
        centSpace.innerHTML = Math.round (centAvg);
        if (centAvg >= 0) {
            centSpace.style.width = '50%';
            centMeter.style.width = centAvg + '%';
        } else {
            centSpace.style.width = (50 + centAvg) + '%';
            centMeter.style.width = -centAvg + '%';
        }
        levMeter.style.background = '#cc0';
        nootElem.style.background = '#cc0';
    } else {
        var note_letter = '---'
        prev_period = 1;
        levMeter.style.background = 'orange';
        nootElem.style.background = 'none';
    }

    blauweLijn ();  // clear + blauwe streep als markering van de sensitivity
    graaf.strokeStyle = 'green';
    graaf.beginPath ();
    graaf.moveTo (0, 256 - diffs [0] * 128);
    var step = graaf_breedte / 512;
    for (var i = 1; i < graaf_breedte; i += step) {
        graaf.lineTo (i / step, 256 - diffs [i] * 128);
    }
    graaf.stroke ();
    graaf.beginPath (); // rood streepje als markering van de periode
    graaf.moveTo (p.period / step, 256);
    graaf.strokeStyle = 'red';
    graaf.lineTo (p.period / step, 245);
    graaf.stroke ();

    var db = 20 * Math.log10 (p.max);   // 20 * log10 (dv) => dB
    levMax = (40 + db) * 2.5;           // -40db => levMax == 0, 0dB => levMax == 100
    if (levMax > uitslag) {
        uitslag = levMax;
        levMeter.innerHTML = db.toFixed (0);
        frameMax = frame.slice ();      // ondiepe kopie
    } else uitslag -= 1;
    levMeter.style.width = uitslag + '%';

    if (dt > maxDt) maxDt = dt;
    if (lusTel++ == 20) {
        tijdMeter.style.width = Math.min (dt * 10, 100) + '%';
        tijdMeter.innerHTML = dt;
        lusTel = maxDt = 0;
    }

    if (frameMax) {
        graaf.strokeStyle = 'grey';
        graaf.beginPath ();
        graaf.moveTo (0, 128 - frameMax [0] * 32);
        var step = graaf_breedte / 512;
        for (var i = 1; i < graaf_breedte; i += step) {
            graaf.lineTo (i / step, 128 - frameMax [i] * 32);
        }
        graaf.stroke ();
    }

	rafID = requestAnimationFrame (function () { updatePitch (1); });
}

function blauweLijn () {
    graaf.clearRect (0, 0, 512, 256);
    graaf.beginPath (); // blauwe streep als markering van de sensitivity
    graaf.moveTo (0, 256 - drempel * 128);
    graaf.strokeStyle = 'blue';
    graaf.setLineDash([5, 3]);
    graaf.lineTo (512, 256 - drempel * 128);
    graaf.stroke ();
    graaf.setLineDash ([]);
}

function setFullscreen () {
    var e = document.body;
    var fscrAan = e.requestFullscreen || e.mozRequestFullScreen || e.webkitRequestFullscreen;
    var fscrUit = document.exitFullscreen || document.mozCancelFullScreen || document.webkitExitFullscreen;
    if (!fscrAan || !fscrUit) return;
    if ($('#fscr').prop ('checked')) fscrAan.call (e);
    else fscrUit.call (document);
}
const MIDI_SERVICE_UID = "03b80e5a-ede8-4b33-a751-6ce34ec4c700";
const MIDI_CHAR_DATAIO = "7772e5db-3868-4112-a1a9-f2669d106bf3";
function scanBLE () {
    var mididataio, fp = { filters: [{ services: [MIDI_SERVICE_UID] }] };
    navigator.bluetooth.requestDevice (fp)
    .then (device => {
        var serv = device.gatt.connect ();
        logerr ('BLE Connected: ' + device.name + ', ' + device.id.substring (0,5) + '...');
        return serv;
    })
    .then (server => server.getPrimaryService (MIDI_SERVICE_UID))
    .then (service => service.getCharacteristic (MIDI_CHAR_DATAIO))
    .then (characteristic => {
        mididataio = characteristic;
        return mididataio.startNotifications ();
    })
    .then (_ => {
        logerr ('Midi notifications started');
        mididataio.addEventListener ('characteristicvaluechanged', handleNotifications);
    })
    .catch (error => { logerr (error.toString ()); });
};

function handleNotifications (event) {
    var chan = event.target.value.getUint8 (2);
    var note = event.target.value.getUint8 (3);
    var vel = event.target.value.getUint8 (4);
    echoMIDIMessage ({ data: [chan, note, vel] });
}

function waitDlg (id, show, allowEnable = 1) {
    if (show) {
        $(id).toggle (true);
        nAsyncLoads += 1;
        startKnop.disabled = true;
    } else {
        $(id).toggle (false); 
        nAsyncLoads -= 1;
        if (nAsyncLoads == 0 && allowEnable) startKnop.disabled = false;
    }
}

function addTips () {
    $('#kwart').attr ('title','Click here to start and keep the metronome on.\nFlashes and clicks on each beat. First beat extra loud');
    $('#tempo').attr ('title','Tempo of the metronome in beats per minute (bpm)');
    $('#start').attr ('title','Start/stop playing.');
    $('#micon').attr ('title','Switches the microphone on/off.');
    $('#vol').attr ('title','The volume of the internal synthesizer');
    $('#error').attr ('title','Shows how correct you have played.\nReaches 100% at the end of a correctly played score.');
    var x = ['Playback: plays the score and checks notes if you play them.',
        'Constant tempo: plays the score, but waits until you play the correct notes.',
        'Adaptive: Adapts the tempo to your playing. It can speed up or slow down (until the given minimum tempo).'
    ].join ('\n');
    $('#l0').attr ('title', x);
    $('#l1').attr ('title','Select which staves you want to play (and checked while playing).');
    $('#flbl').attr ('title','Load a score file from your local file system. (Or an audio file or a preload file.)');
    $('#drplbl').attr ('title','Load the score file from your Dropbox.');
    $('#l2').attr ('title','Use internal synthesizer or MIDI interface or no sound at all.');
    $('#l3').attr ('title','Also display a line cursor that\nmoves continuously. (with score time)');
    $('#l4').attr ('title','Hide the normal note cursors.');
    $('#l5').attr ('title','Restrict the play area with two loop marks.\nTo move a mark: first select then click new position');
    $('#stoplbl').attr ('title','When end of loop is reached, stop playing\nand jump to loop begin.');
    $('#l6').attr ('title','Display the score in one long line.\nThe line scrolls horizontally when playing.');
    $('#pwlbl').attr ('title','Sets the score page width in centimeters.\nZero resets to default.\nUseful to enlarge the score.');
    $('#scllbl').attr ('title','Scale factor for side scrolling score.');
    $('#l7').attr ('title','Keep the metronome ticking after the count-in.');
    $('#l8').attr ('title','When the score has lyrics, hide it');
    $('#l9').attr ('title','Do not synthesize the staff(s) you are playing yourself');
    $('#lA').attr ('title','Only synthesize the staff(s) you are\nplaying yourself (and mute all others)');
    $('#lB').attr ('title','Show the on-screen keyboard.');
    $('#marklbl').attr ('title','Show names of the notes highlighted by the cursors');
    $('#lC').attr ('title','Only show the selected staff (part) from a concert score');
    $('#lD').attr ('title','Connect to a MIDI device via Bluetooth LE.');
    $('#lE').attr ('title','Show the advanced menu');
    $('#lF').attr ('title','Show a graph of the performance\nThe timing errors of all played notes and a runnung average of 10 notes');
    $('#lG').attr ('title','Check if notes are held down for the right duration.\nMark notes (orange) played too short (or long)');
    $('#lH').attr ('title','Check if notes are played at the right time.\nMark notes (gold) played too late.');
    $('#lI').attr ('title','Show permanent error marks and timing marks');
    $('#lJ').attr ('title','Start without count-in');
    $('#').attr ('title','');
}

$(document).ready (function () {
    audElm = document.getElementById ('aud');
    volElm = document.getElementById ('vol') || { addEventListener: () => {} }; // dummy object als schuifregelaar niet bestaat
    perfElm = document.getElementById ('perf');
    deNot = document.getElementById ('notation');
    hasSmooth = CSS.supports ('scroll-behavior','smooth');
    startKnop = document.getElementById ('start');
    $('#verlab').html ('Version: ' + follow_VERSION);
    $('body').keydown (keyDown)
             .click (function () { if (menuVisible) toggleMenu (); });
    $('#perfbtn').click (evt => { evt.stopPropagation (); perfElm.style.display = 'none'; });
    perfElm.addEventListener ('click', evt => { evt.stopPropagation (); }); // klik niet naar body
    //------------- knoppen in de balk
    $('#kwart').click (function () {
        opt.metro = 1;
        $('#metro').prop ('checked', 1)
        toggleStart (-1);
    });
    $('#kwart').append ($('#mtrsvg'));
    $('#tempo').change (function () {
        kwartTempo = $tempo.value * (curNote.met / 384);
        opt.tempo = kwartTempo;
        setAudSpeed ();
    });
    $('#micon').click (toggleMic);
    $('#start').click (function () { toggleStart (-2); });
    $('#spkr').append ($('#spkron'));
    $('#spkr').append ($('#spkrof').toggle (false));
    $('#spkr').toggle (false).click (function (evt) {evt.stopPropagation (); schakelAudio (!opt.audOn);});
    $('#menu').toggle (false);   // verberg menu in het begin
    menuVisible = 0;
    $('#menu').click (function (ev) { ev.stopPropagation (); });
    $('#mbar').click (function (ev) { ev.stopPropagation (); showMenu (1); });
    $('#mbar2').click (function (ev) { ev.stopPropagation (); showMenu (2); });
    $('#mbar3').click (function (ev) { ev.stopPropagation (); showMenu (3); });
    //------------- checkbox en input
    $('#menu select').change (checkMenu);
    $('#menu input').change (checkMenu);
    $('#tuner input').change (checkMenu);
    //------------- grootte veranderen, schuiven e.d.
    $('#zoom').on ('mousedown touchstart', zoomToetsen);
    $('#rollijn').on ('mousedown touchstart', lijn_shift);
    $('#info').on ('mousedown touchstart', zoomInfo);
    $(window).resize (function () {
        setYSvgs ();
        setScale ();
        resizeNotation ();
        alignSystem ();
    });
    //------------- knoppen in het menu
    $('#fknp').change (readLocalFile);
    $('#testport').on ('mousedown', testPort);
    $('#stats').click (function () { showStats (0); });
    $('#save').click (savePreload);
    $('#saveok').click (function () { $('#saveDlg').toggle (); });
    $('#micKnop').click (function () { showMicDlg (true); })
    $('#ble').click (scanBLE);
    //------------- voor de mikrofoondialoog
    $('#micOk').click (function () { showMicDlg (false); });
    levMeter = document.querySelector ('#mxlvl > div');  levMeter.innerHTML = '0';
    centMeter = document.querySelector ('#cents > div'); centMeter.innerHTML = '&nbsp;';
    centSpace = document.querySelector ('#cents > span'); centSpace.style.width = '50%';
    nootElem = document.getElementById ('noot');
    tijdMeter = document.querySelector ('#anatijd > div');  tijdMeter.innerHTML = '0';
    graaf = document.getElementById ('graaf').getContext ('2d');
    //-------------
    verbindSchuifknop ('#kblft');
    verbindSchuifknop ('#kbrgt');
    $error = $('#error')[0];
    $tempo = $('#tempo')[0]; $mtrpad = $('#path9868')[0];
    atag = document.getElementById ('atag'); atag.style.display = 'none';
    btag = document.getElementById ('btag'); btag.style.display = 'none';
    clearTiming (1);
    var ac = window.AudioContext || window.webkitAudioContext;
    audioCtx = ac != undefined ? new ac () : null;
    if (!audioCtx) alert ('Your browser does not support the Web Audio API');
    decodeer (E3, function (data) {
        metroGolf = data;
        console.log ('E3 decoded');
    });
    setFHoog ();
    $('#err').text ('');
    initOptions ();
    tryMidi ();
    //~ audioCtx.suspend ().then (tryMidi); // debug new policy
    maakToetsen ();
    addTips ();
    $('#adv, .ext').toggle ($('#advanced').prop ('checked')); // maak zichtbaarheid consistent met checkboxstatus (firefox reload)
    $('#advanced').on ('change', () => $('#adv, .ext').toggle ($('#advanced').prop ('checked')));
    $('#fscr').on ('change', setFullscreen);
    $('body').on ('fullscreenchange webkitfullscreenchange mozfullscreenchange', function () {
        var e = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement;
        $('#fscr').prop ('checked', e != null);
    });
    deNot.addEventListener ('touchstart', function (evt) {
        xScrollStart = evt.touches[0].clientX
    });
    deNot.addEventListener ('touchmove', function (evt) {
        if (startKnop.value == 'stop') toggleStart (iSeq);  // scrolling -> press stop button
        if (opt.hrz) {  // do our own x scrolling to avoid bug with iOS and overflow-y:hidden
            evt.stopPropagation (); evt.preventDefault ();
            var x = evt.touches[0].clientX
            deNot.scrollLeft -= x - xScrollStart;
            xScrollStart = evt.touches[0].clientX
        }
    });
});
})();

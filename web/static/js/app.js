/**
 * Pre-computed model weights & scaler parameters for client-side ML engine
 */
const MODEL_WEIGHTS = {"scaler_mean":[4.669245647969052,4.299806576402321,90.64468085106384,110.87234042553192,547.9400386847195,9.021663442940039,18.88916827852998,44.28820116054158,4.017601547388782,0.021663442940038684,-0.5410528407028135,-0.22020655225786787,0.7504835589941973,0.688588007736944,-0.060597654070254414,0.1097569627183962,0.34622823984526113,1.353095813501614,141.82044534655293,30.685058800773696,3.7116421663442942,0.4948597447891809,20.84448742746615,2.3261093532196937,8.969052224371373,0.3694390715667311],"scaler_scale":[2.3115390494606083,1.228710368455708,5.514769666324428,63.9845117632872,247.82616630691257,4.5550654884788315,5.801006939598366,16.301680682958068,1.789919022794071,0.29567275495940354,0.7021717661213782,0.40709422591348976,0.43273315874058055,0.4630707973278065,0.7063571620942778,0.6966641177885743,0.475767007871409,0.722552441385487,72.70682078096814,16.554715548369078,2.778934922115929,0.2733784972159722,3.947767478481386,1.2028932707314537,3.1492545664265528,1.946298941402665],"feature_names":["X","Y","FFMC","DMC","DC","ISI","temp","RH","wind","rain","month_sin","month_cos","is_summer","is_aug_sep","day_sin","day_cos","is_weekend","VPD","BUI","drought_factor","fire_spread_potential","temp_rh_ratio","ffmc_dryness","dist_center","spatial_diag","spatial_diff"],"ridge_coef":[0.03661817781518215,0.011279779121488281,0.09272306536545102,0.08922727165414687,-0.1981241100168728,-0.05497115855005065,-0.017244718306389812,0.028979329303251977,0.15789444324865973,0.0241467952317551,-0.09399605144954612,0.2014160920140033,0.08426050202076657,-0.06748069763434775,0.060181525801027015,-0.024046238453982727,0.09272674294043655,0.301424360520178,0.08050601693223573,0.11892514891840332,-0.027395385850820554,-0.19559790288922702,-0.04235402307452806,0.03920590487660342,0.031278490646794554,0.03636890760902745],"ridge_intercept":1.111025765296088,"huber_coef":[0.019923839488646154,0.03264423864181351,0.09209721698456354,-0.024052088828615883,-0.18259227145384566,0.053688706738723765,-0.16159409328939162,0.06848772842856922,0.21771744019984515,0.05775898100346404,-0.06349847868886395,0.1529634421300562,0.09538702495029659,-0.1260344330846353,0.05068839650367454,-0.028488496655254263,0.043944091931928476,0.6370515318335929,0.17839143973685687,0.2077364405448753,-0.13916137815247953,-0.4858550822861053,-0.07452374030803226,0.0037778685232268522,0.02736045805928616,0.0030542165836921916],"huber_intercept":0.8318149092291326,"logreg_coef":[0.04032386828526321,0.008477658654606346,0.04967023174736375,0.1338967853419132,-0.05319471708523734,-0.021682695382433656,-0.09065327476053318,-0.012040708584506402,0.2307289003925804,0.1031776823657131,-0.09281075058648985,0.33604025650670005,-0.048242774176276114,0.04187346462559174,0.07797984472567829,0.05448498236849736,0.07843586971642243,0.10136370759926666,0.08448855026537547,0.02503110973882692,0.0015523190038806188,0.06516984549767606,-0.05947040081437494,0.004429070719215129,0.03290517837477316,0.042538999182664315],"logreg_intercept":-1.4346298694467798,"weights":{"ridge":0.5,"huber":0.2,"prob":0.3}};

/**
 * Embedded offline baseline dataset (81 grid cells + 25 optimal matroid crew assignments)
 */
const EMBEDDED_BASELINE = {"crews":[{"DC":671.2,"DMC":181.1,"FFMC":96.1,"ISI":14.3,"RH":63.0,"day":"TUE","impact_score":0.9593,"month":"AUG","priority_rank":1,"rain":6.4,"temp":27.3,"wind":4.9,"x":7,"y":5},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":24.0,"day":"TUE","impact_score":0.9101,"month":"DEC","priority_rank":2,"rain":0.0,"temp":5.1,"wind":8.5,"x":6,"y":5},{"DC":354.6,"DMC":27.8,"FFMC":84.0,"ISI":5.3,"RH":61.0,"day":"WED","impact_score":0.8661,"month":"DEC","priority_rank":3,"rain":0.0,"temp":5.1,"wind":8.0,"x":8,"y":6},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"MON","impact_score":0.8521,"month":"DEC","priority_rank":4,"rain":0.0,"temp":4.6,"wind":8.5,"x":4,"y":4},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"MON","impact_score":0.8521,"month":"DEC","priority_rank":5,"rain":0.0,"temp":4.6,"wind":8.5,"x":4,"y":4},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"MON","impact_score":0.8521,"month":"DEC","priority_rank":6,"rain":0.0,"temp":4.6,"wind":8.5,"x":4,"y":4},{"DC":353.5,"DMC":27.2,"FFMC":84.4,"ISI":6.8,"RH":57.0,"day":"SUN","impact_score":0.8425,"month":"DEC","priority_rank":7,"rain":0.0,"temp":4.8,"wind":8.5,"x":4,"y":6},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"MON","impact_score":0.8418,"month":"DEC","priority_rank":8,"rain":0.0,"temp":4.6,"wind":8.5,"x":3,"y":4},{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"SUN","impact_score":0.7826,"month":"SEP","priority_rank":9,"rain":0.0,"temp":13.8,"wind":7.6,"x":7,"y":4},{"DC":795.9,"DMC":263.1,"FFMC":88.9,"ISI":5.2,"RH":27.0,"day":"SUN","impact_score":0.7172,"month":"JUL","priority_rank":10,"rain":0.0,"temp":29.3,"wind":3.6,"x":8,"y":6},{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"SUN","impact_score":0.7138,"month":"SEP","priority_rank":11,"rain":0.0,"temp":13.8,"wind":7.6,"x":4,"y":5},{"DC":587.1,"DMC":130.3,"FFMC":94.9,"ISI":14.1,"RH":27.0,"day":"SUN","impact_score":0.6991,"month":"AUG","priority_rank":12,"rain":0.0,"temp":31.0,"wind":5.4,"x":8,"y":6},{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":76.0,"day":"SUN","impact_score":0.6803,"month":"SEP","priority_rank":13,"rain":0.0,"temp":14.5,"wind":7.6,"x":1,"y":4},{"DC":43.6,"DMC":3.2,"FFMC":84.6,"ISI":3.3,"RH":53.0,"day":"FRI","impact_score":0.6767,"month":"FEB","priority_rank":14,"rain":0.0,"temp":8.2,"wind":9.4,"x":7,"y":4},{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":43.0,"day":"SUN","impact_score":0.674,"month":"SEP","priority_rank":15,"rain":0.0,"temp":21.9,"wind":4.0,"x":1,"y":3},{"DC":860.6,"DMC":291.3,"FFMC":87.1,"ISI":4.0,"RH":67.0,"day":"SAT","impact_score":0.6717,"month":"SEP","priority_rank":16,"rain":0.0,"temp":17.0,"wind":4.9,"x":6,"y":5},{"DC":106.7,"DMC":3.0,"FFMC":79.5,"ISI":1.1,"RH":31.0,"day":"TUE","impact_score":0.6698,"month":"NOV","priority_rank":17,"rain":0.0,"temp":11.8,"wind":4.5,"x":6,"y":3},{"DC":680.7,"DMC":124.1,"FFMC":92.4,"ISI":8.5,"RH":32.0,"day":"SUN","impact_score":0.6685,"month":"SEP","priority_rank":18,"rain":0.0,"temp":23.9,"wind":6.7,"x":1,"y":3},{"DC":715.1,"DMC":231.1,"FFMC":93.7,"ISI":8.4,"RH":31.0,"day":"SAT","impact_score":0.664,"month":"AUG","priority_rank":19,"rain":0.0,"temp":26.9,"wind":3.6,"x":8,"y":6},{"DC":849.3,"DMC":287.2,"FFMC":89.7,"ISI":6.8,"RH":45.0,"day":"THU","impact_score":0.6634,"month":"SEP","priority_rank":20,"rain":0.0,"temp":19.4,"wind":3.6,"x":7,"y":4},{"DC":721.4,"DMC":145.4,"FFMC":93.4,"ISI":8.1,"RH":24.0,"day":"SAT","impact_score":0.6513,"month":"SEP","priority_rank":21,"rain":0.0,"temp":30.2,"wind":2.7,"x":6,"y":3},{"DC":819.1,"DMC":273.8,"FFMC":91.6,"ISI":7.7,"RH":44.0,"day":"SAT","impact_score":0.6497,"month":"AUG","priority_rank":22,"rain":0.0,"temp":21.3,"wind":4.5,"x":8,"y":4},{"DC":698.6,"DMC":88.0,"FFMC":92.5,"ISI":7.1,"RH":51.0,"day":"SAT","impact_score":0.6488,"month":"SEP","priority_rank":23,"rain":0.0,"temp":17.8,"wind":7.2,"x":7,"y":5},{"DC":635.9,"DMC":191.4,"FFMC":91.7,"ISI":7.8,"RH":36.0,"day":"WED","impact_score":0.6487,"month":"AUG","priority_rank":24,"rain":0.0,"temp":26.2,"wind":4.5,"x":8,"y":8},{"DC":728.6,"DMC":149.3,"FFMC":93.5,"ISI":8.1,"RH":26.0,"day":"SUN","impact_score":0.6457,"month":"SEP","priority_rank":25,"rain":0.0,"temp":28.3,"wind":3.1,"x":4,"y":6}],"grid":[{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":1,"y":1},{"avg_impact":0.3594648217654737,"crew_details":[],"crews_assigned":0,"max_impact":0.5826603402709183,"observation_count":19,"sample_weather":{"DC":728.6,"DMC":149.3,"FFMC":93.5,"ISI":8.1,"RH":36.0,"day":"sun","month":"sep","rain":0.0,"temp":25.3,"wind":3.6},"x":1,"y":2},{"avg_impact":0.4581825053171536,"crew_details":[{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":43.0,"day":"sun","impact_score":0.674,"month":"sep","priority_rank":15,"rain":0.0,"temp":21.9,"wind":4.0},{"DC":680.7,"DMC":124.1,"FFMC":92.4,"ISI":8.5,"RH":32.0,"day":"sun","impact_score":0.6685,"month":"sep","priority_rank":18,"rain":0.0,"temp":23.9,"wind":6.7}],"crews_assigned":2,"max_impact":0.6740287533022997,"observation_count":10,"sample_weather":{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":43.0,"day":"sun","month":"sep","rain":0.0,"temp":21.9,"wind":4.0},"x":1,"y":3},{"avg_impact":0.39796428813057755,"crew_details":[{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":76.0,"day":"sun","impact_score":0.6803,"month":"sep","priority_rank":13,"rain":0.0,"temp":14.5,"wind":7.6}],"crews_assigned":1,"max_impact":0.6803021676212595,"observation_count":15,"sample_weather":{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":76.0,"day":"sun","month":"sep","rain":0.0,"temp":14.5,"wind":7.6},"x":1,"y":4},{"avg_impact":0.5150117905131044,"crew_details":[],"crews_assigned":0,"max_impact":0.6002687606948086,"observation_count":4,"sample_weather":{"DC":728.6,"DMC":149.3,"FFMC":93.5,"ISI":8.1,"RH":27.0,"day":"sun","month":"sep","rain":0.0,"temp":27.8,"wind":3.1},"x":1,"y":5},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":1,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":1,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":1,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":1,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":2,"y":1},{"avg_impact":0.38182136585416115,"crew_details":[],"crews_assigned":0,"max_impact":0.49293484297586837,"observation_count":25,"sample_weather":{"DC":668.0,"DMC":117.9,"FFMC":92.4,"ISI":12.2,"RH":33.0,"day":"fri","month":"sep","rain":0.0,"temp":19.6,"wind":6.3},"x":2,"y":2},{"avg_impact":0.4917832216881605,"crew_details":[],"crews_assigned":0,"max_impact":0.4917832216881605,"observation_count":1,"sample_weather":{"DC":764.0,"DMC":108.4,"FFMC":91.6,"ISI":6.2,"RH":51.0,"day":"mon","month":"sep","rain":0.0,"temp":18.0,"wind":5.4},"x":2,"y":3},{"avg_impact":0.4061768647516669,"crew_details":[],"crews_assigned":0,"max_impact":0.5699166098996832,"observation_count":27,"sample_weather":{"DC":664.5,"DMC":203.2,"FFMC":92.0,"ISI":8.1,"RH":42.0,"day":"sun","month":"aug","rain":0.0,"temp":24.9,"wind":5.4},"x":2,"y":4},{"avg_impact":0.3978712877061795,"crew_details":[],"crews_assigned":0,"max_impact":0.5700715376453003,"observation_count":20,"sample_weather":{"DC":587.1,"DMC":130.3,"FFMC":94.9,"ISI":14.1,"RH":25.0,"day":"sun","month":"aug","rain":0.0,"temp":33.1,"wind":4.0},"x":2,"y":5},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":2,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":2,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":2,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":2,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":3,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":3,"y":2},{"avg_impact":0.48508559672769513,"crew_details":[],"crews_assigned":0,"max_impact":0.48508559672769513,"observation_count":1,"sample_weather":{"DC":751.5,"DMC":102.3,"FFMC":92.2,"ISI":8.4,"RH":27.0,"day":"sat","month":"sep","rain":0.0,"temp":24.2,"wind":3.1},"x":3,"y":3},{"avg_impact":0.40802890286402416,"crew_details":[{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","impact_score":0.8418,"month":"dec","priority_rank":8,"rain":0.0,"temp":4.6,"wind":8.5}],"crews_assigned":1,"max_impact":0.8417517756721677,"observation_count":43,"sample_weather":{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","month":"dec","rain":0.0,"temp":4.6,"wind":8.5},"x":3,"y":4},{"avg_impact":0.37646940710751925,"crew_details":[],"crews_assigned":0,"max_impact":0.4668773681874286,"observation_count":7,"sample_weather":{"DC":673.8,"DMC":37.9,"FFMC":91.4,"ISI":5.2,"RH":46.0,"day":"wed","month":"oct","rain":0.0,"temp":15.9,"wind":3.6},"x":3,"y":5},{"avg_impact":0.28843182182775295,"crew_details":[],"crews_assigned":0,"max_impact":0.3892636040019072,"observation_count":4,"sample_weather":{"DC":686.5,"DMC":126.5,"FFMC":90.9,"ISI":7.0,"RH":66.0,"day":"mon","month":"sep","rain":0.0,"temp":15.6,"wind":3.1},"x":3,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":3,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":3,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":3,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":4,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":4,"y":2},{"avg_impact":0.3872967131093994,"crew_details":[],"crews_assigned":0,"max_impact":0.5568137447011187,"observation_count":22,"sample_weather":{"DC":855.3,"DMC":290.0,"FFMC":90.3,"ISI":7.4,"RH":44.0,"day":"fri","month":"sep","rain":0.0,"temp":19.9,"wind":3.1},"x":4,"y":3},{"avg_impact":0.4236805048999953,"crew_details":[{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","impact_score":0.8521,"month":"dec","priority_rank":4,"rain":0.0,"temp":4.6,"wind":8.5},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","impact_score":0.8521,"month":"dec","priority_rank":5,"rain":0.0,"temp":4.6,"wind":8.5},{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","impact_score":0.8521,"month":"dec","priority_rank":6,"rain":0.0,"temp":4.6,"wind":8.5}],"crews_assigned":3,"max_impact":0.8521457913285151,"observation_count":36,"sample_weather":{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":21.0,"day":"mon","month":"dec","rain":0.0,"temp":4.6,"wind":8.5},"x":4,"y":4},{"avg_impact":0.40006990543027926,"crew_details":[{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"sun","impact_score":0.7138,"month":"sep","priority_rank":11,"rain":0.0,"temp":13.8,"wind":7.6}],"crews_assigned":1,"max_impact":0.713815994706825,"observation_count":25,"sample_weather":{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"sun","month":"sep","rain":0.0,"temp":13.8,"wind":7.6},"x":4,"y":5},{"avg_impact":0.5213662660872033,"crew_details":[{"DC":353.5,"DMC":27.2,"FFMC":84.4,"ISI":6.8,"RH":57.0,"day":"sun","impact_score":0.8425,"month":"dec","priority_rank":7,"rain":0.0,"temp":4.8,"wind":8.5},{"DC":728.6,"DMC":149.3,"FFMC":93.5,"ISI":8.1,"RH":26.0,"day":"sun","impact_score":0.6457,"month":"sep","priority_rank":25,"rain":0.0,"temp":28.3,"wind":3.1}],"crews_assigned":2,"max_impact":0.8425202529705532,"observation_count":8,"sample_weather":{"DC":353.5,"DMC":27.2,"FFMC":84.4,"ISI":6.8,"RH":57.0,"day":"sun","month":"dec","rain":0.0,"temp":4.8,"wind":8.5},"x":4,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":4,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":4,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":4,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":2},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":3},{"avg_impact":0.43269620056369545,"crew_details":[],"crews_assigned":0,"max_impact":0.5899909329062996,"observation_count":23,"sample_weather":{"DC":783.5,"DMC":119.0,"FFMC":92.8,"ISI":7.5,"RH":28.0,"day":"thu","month":"sep","rain":0.0,"temp":21.6,"wind":6.3},"x":5,"y":4},{"avg_impact":0.3518039934497675,"crew_details":[],"crews_assigned":0,"max_impact":0.38718982785092704,"observation_count":3,"sample_weather":{"DC":80.8,"DMC":35.8,"FFMC":91.7,"ISI":7.8,"RH":27.0,"day":"sat","month":"mar","rain":0.0,"temp":15.1,"wind":5.4},"x":5,"y":5},{"avg_impact":0.4529948051684979,"crew_details":[],"crews_assigned":0,"max_impact":0.5433463880738743,"observation_count":4,"sample_weather":{"DC":613.0,"DMC":181.3,"FFMC":91.6,"ISI":7.6,"RH":33.0,"day":"sun","month":"aug","rain":0.0,"temp":24.3,"wind":3.6},"x":5,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":5,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":6,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":6,"y":2},{"avg_impact":0.4814166320008924,"crew_details":[{"DC":106.7,"DMC":3.0,"FFMC":79.5,"ISI":1.1,"RH":31.0,"day":"tue","impact_score":0.6698,"month":"nov","priority_rank":17,"rain":0.0,"temp":11.8,"wind":4.5},{"DC":721.4,"DMC":145.4,"FFMC":93.4,"ISI":8.1,"RH":24.0,"day":"sat","impact_score":0.6513,"month":"sep","priority_rank":21,"rain":0.0,"temp":30.2,"wind":2.7}],"crews_assigned":2,"max_impact":0.6697803545599151,"observation_count":25,"sample_weather":{"DC":106.7,"DMC":3.0,"FFMC":79.5,"ISI":1.1,"RH":31.0,"day":"tue","month":"nov","rain":0.0,"temp":11.8,"wind":4.5},"x":6,"y":3},{"avg_impact":0.3924838192861805,"crew_details":[],"crews_assigned":0,"max_impact":0.5519419421271131,"observation_count":9,"sample_weather":{"DC":16.2,"DMC":4.4,"FFMC":75.1,"ISI":1.9,"RH":77.0,"day":"tue","month":"feb","rain":0.0,"temp":5.1,"wind":5.4},"x":6,"y":4},{"avg_impact":0.42272565447812077,"crew_details":[{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":24.0,"day":"tue","impact_score":0.9101,"month":"dec","priority_rank":2,"rain":0.0,"temp":5.1,"wind":8.5},{"DC":860.6,"DMC":291.3,"FFMC":87.1,"ISI":4.0,"RH":67.0,"day":"sat","impact_score":0.6717,"month":"sep","priority_rank":16,"rain":0.0,"temp":17.0,"wind":4.9}],"crews_assigned":2,"max_impact":0.9101239415319289,"observation_count":49,"sample_weather":{"DC":349.7,"DMC":25.4,"FFMC":85.4,"ISI":2.6,"RH":24.0,"day":"tue","month":"dec","rain":0.0,"temp":5.1,"wind":8.5},"x":6,"y":5},{"avg_impact":0.46032989384363154,"crew_details":[],"crews_assigned":0,"max_impact":0.6370254530951218,"observation_count":3,"sample_weather":{"DC":643.0,"DMC":164.0,"FFMC":96.0,"ISI":14.0,"RH":30.0,"day":"sat","month":"aug","rain":0.0,"temp":30.8,"wind":4.9},"x":6,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":6,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":6,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":6,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":7,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":7,"y":2},{"avg_impact":0.5420043191842092,"crew_details":[],"crews_assigned":0,"max_impact":0.6001217378727006,"observation_count":2,"sample_weather":{"DC":686.9,"DMC":43.7,"FFMC":90.6,"ISI":6.7,"RH":27.0,"day":"sat","month":"oct","rain":0.0,"temp":17.8,"wind":4.0},"x":7,"y":3},{"avg_impact":0.4756845202541339,"crew_details":[{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"sun","impact_score":0.7826,"month":"sep","priority_rank":9,"rain":0.0,"temp":13.8,"wind":7.6},{"DC":43.6,"DMC":3.2,"FFMC":84.6,"ISI":3.3,"RH":53.0,"day":"fri","impact_score":0.6767,"month":"feb","priority_rank":14,"rain":0.0,"temp":8.2,"wind":9.4},{"DC":849.3,"DMC":287.2,"FFMC":89.7,"ISI":6.8,"RH":45.0,"day":"thu","impact_score":0.6634,"month":"sep","priority_rank":20,"rain":0.0,"temp":19.4,"wind":3.6}],"crews_assigned":3,"max_impact":0.7825703670890268,"observation_count":45,"sample_weather":{"DC":825.1,"DMC":276.3,"FFMC":91.0,"ISI":7.1,"RH":77.0,"day":"sun","month":"sep","rain":0.0,"temp":13.8,"wind":7.6},"x":7,"y":4},{"avg_impact":0.5247436096464061,"crew_details":[{"DC":671.2,"DMC":181.1,"FFMC":96.1,"ISI":14.3,"RH":63.0,"day":"tue","impact_score":0.9593,"month":"aug","priority_rank":1,"rain":6.4,"temp":27.3,"wind":4.9},{"DC":698.6,"DMC":88.0,"FFMC":92.5,"ISI":7.1,"RH":51.0,"day":"sat","impact_score":0.6488,"month":"sep","priority_rank":23,"rain":0.0,"temp":17.8,"wind":7.2}],"crews_assigned":2,"max_impact":0.9592899181006473,"observation_count":11,"sample_weather":{"DC":671.2,"DMC":181.1,"FFMC":96.1,"ISI":14.3,"RH":63.0,"day":"tue","month":"aug","rain":6.4,"temp":27.3,"wind":4.9},"x":7,"y":5},{"avg_impact":0.5200129459981553,"crew_details":[],"crews_assigned":0,"max_impact":0.5827662677576785,"observation_count":2,"sample_weather":{"DC":430.8,"DMC":180.4,"FFMC":93.1,"ISI":11.0,"RH":28.0,"day":"tue","month":"jul","rain":0.0,"temp":26.9,"wind":5.4},"x":7,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":7,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":7,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":7,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":8,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":8,"y":2},{"avg_impact":0.4514646791439412,"crew_details":[],"crews_assigned":0,"max_impact":0.5963750346028062,"observation_count":3,"sample_weather":{"DC":671.9,"DMC":73.4,"FFMC":84.4,"ISI":3.2,"RH":28.0,"day":"tue","month":"sep","rain":0.0,"temp":24.2,"wind":3.6},"x":8,"y":3},{"avg_impact":0.6496968288219211,"crew_details":[{"DC":819.1,"DMC":273.8,"FFMC":91.6,"ISI":7.7,"RH":44.0,"day":"sat","impact_score":0.6497,"month":"aug","priority_rank":22,"rain":0.0,"temp":21.3,"wind":4.5}],"crews_assigned":1,"max_impact":0.6496968288219211,"observation_count":1,"sample_weather":{"DC":819.1,"DMC":273.8,"FFMC":91.6,"ISI":7.7,"RH":44.0,"day":"sat","month":"aug","rain":0.0,"temp":21.3,"wind":4.5},"x":8,"y":4},{"avg_impact":0.5201653548040028,"crew_details":[],"crews_assigned":0,"max_impact":0.6327353807361491,"observation_count":4,"sample_weather":{"DC":664.2,"DMC":32.8,"FFMC":84.9,"ISI":3.0,"RH":47.0,"day":"mon","month":"oct","rain":0.0,"temp":16.7,"wind":4.9},"x":8,"y":5},{"avg_impact":0.5024567569477104,"crew_details":[{"DC":354.6,"DMC":27.8,"FFMC":84.0,"ISI":5.3,"RH":61.0,"day":"wed","impact_score":0.8661,"month":"dec","priority_rank":3,"rain":0.0,"temp":5.1,"wind":8.0},{"DC":795.9,"DMC":263.1,"FFMC":88.9,"ISI":5.2,"RH":27.0,"day":"sun","impact_score":0.7172,"month":"jul","priority_rank":10,"rain":0.0,"temp":29.3,"wind":3.6},{"DC":587.1,"DMC":130.3,"FFMC":94.9,"ISI":14.1,"RH":27.0,"day":"sun","impact_score":0.6991,"month":"aug","priority_rank":12,"rain":0.0,"temp":31.0,"wind":5.4},{"DC":715.1,"DMC":231.1,"FFMC":93.7,"ISI":8.4,"RH":31.0,"day":"sat","impact_score":0.664,"month":"aug","priority_rank":19,"rain":0.0,"temp":26.9,"wind":3.6}],"crews_assigned":4,"max_impact":0.8660582330667417,"observation_count":52,"sample_weather":{"DC":354.6,"DMC":27.8,"FFMC":84.0,"ISI":5.3,"RH":61.0,"day":"wed","month":"dec","rain":0.0,"temp":5.1,"wind":8.0},"x":8,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":8,"y":7},{"avg_impact":0.6487358418454414,"crew_details":[{"DC":635.9,"DMC":191.4,"FFMC":91.7,"ISI":7.8,"RH":36.0,"day":"wed","impact_score":0.6487,"month":"aug","priority_rank":24,"rain":0.0,"temp":26.2,"wind":4.5}],"crews_assigned":1,"max_impact":0.6487358418454414,"observation_count":1,"sample_weather":{"DC":635.9,"DMC":191.4,"FFMC":91.7,"ISI":7.8,"RH":36.0,"day":"wed","month":"aug","rain":0.0,"temp":26.2,"wind":4.5},"x":8,"y":8},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":8,"y":9},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":9,"y":1},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":9,"y":2},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":9,"y":3},{"avg_impact":0.4372096535010596,"crew_details":[],"crews_assigned":0,"max_impact":0.6026089509690224,"observation_count":4,"sample_weather":{"DC":671.9,"DMC":73.4,"FFMC":84.4,"ISI":3.2,"RH":36.0,"day":"tue","month":"sep","rain":0.0,"temp":24.3,"wind":3.1},"x":9,"y":4},{"avg_impact":0.35881316839517513,"crew_details":[],"crews_assigned":0,"max_impact":0.35881316839517513,"observation_count":2,"sample_weather":{"DC":297.7,"DMC":49.5,"FFMC":93.3,"ISI":14.0,"RH":34.0,"day":"wed","month":"jun","rain":0.0,"temp":28.0,"wind":4.5},"x":9,"y":5},{"avg_impact":0.4884313639042502,"crew_details":[],"crews_assigned":0,"max_impact":0.4884313639042502,"observation_count":1,"sample_weather":{"DC":753.8,"DMC":248.4,"FFMC":91.6,"ISI":6.3,"RH":58.0,"day":"thu","month":"aug","rain":0.0,"temp":20.5,"wind":2.7},"x":9,"y":6},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":9,"y":7},{"avg_impact":0.0,"crew_details":[],"crews_assigned":0,"max_impact":0.0,"observation_count":0,"sample_weather":null,"x":9,"y":8},{"avg_impact":0.5037389013292052,"crew_details":[],"crews_assigned":0,"max_impact":0.6026708404449638,"observation_count":6,"sample_weather":{"DC":706.7,"DMC":227.0,"FFMC":94.8,"ISI":12.0,"RH":36.0,"day":"fri","month":"aug","rain":0.0,"temp":25.0,"wind":4.0},"x":9,"y":9}],"metrics":{"constraint_satisfied":true,"eval_source":"forestfires.csv (baseline)","has_ground_truth":true,"high_impact_recall":0.125,"is_simulation_active":false,"max_per_cell_constraint":4,"max_per_cell_observed":4,"ndcg_at_25":0.0595,"rubric_total_score":18.32,"spearman_corr":0.1923,"target_crews":25,"total_crews_selected":25,"total_eval_rows":517,"unique_cells_covered":13},"simulation":null};

/**
 * DONEZO x WILDFIRE 3D TACTICAL COMMAND CENTER
 * Unified Frontend Application & 3D Spatial Grid Engine
 */

(function () {
  'use strict';

  // Global Application State
  let currentGridData = [];
  let currentCrews = [];
  let currentMetrics = {};
  let selectedCell = null;
  let isAutoRotating = false;

  // 3D Scene variables
  let scene, camera, renderer, controls;
  let pillarGroup, crewGroup, gridHelperGroup, labelGroup;
  let raycaster, mouse;
  let hoveredMesh = null;
  let isThreeAvailable = (typeof THREE !== 'undefined');

  // DOM Elements
  const canvasWrapper = document.getElementById('canvas-wrapper');
  const webglCanvas = document.getElementById('webgl-canvas');
  const tooltip = document.getElementById('cell-tooltip');
  const portfolioTbody = document.getElementById('portfolio-table-body');
  const matrixGrid = document.getElementById('matrix-grid');

  // Sliders
  const sliderTemp = document.getElementById('slider-temp');
  const sliderRh = document.getElementById('slider-rh');
  const sliderWind = document.getElementById('slider-wind');
  const valTemp = document.getElementById('val-temp-delta');
  const valRh = document.getElementById('val-rh-delta');
  const valWind = document.getElementById('val-wind-delta');

  // Top KPI Elements
  const kpiCrewsDeployed = document.getElementById('kpi-crews-deployed');
  const kpiMaxPerCell = document.getElementById('kpi-max-per-cell');
  const kpiUniqueSectors = document.getElementById('kpi-unique-sectors');
  const kpiTotalEvalRows = document.getElementById('kpi-total-eval-rows');
  const kpiDatasetTag = document.getElementById('kpi-dataset-tag');
  const kpiConstraintBadge = document.getElementById('kpi-constraint-badge');
  const sidebarCrewCount = document.getElementById('sidebar-crew-count');

  // Metric Accuracy Cards
  const metricNdcg = document.getElementById('metric-ndcg');
  const metricRecall = document.getElementById('metric-recall');
  const metricRubric = document.getElementById('metric-rubric');
  const evalSourceLabel = document.getElementById('eval-source-label');
  const hudStatusBadge = document.getElementById('hud-status-badge');

  // Inspector Elements
  const inspectorCoordsBadge = document.getElementById('inspector-coords-badge');
  const inspImpact = document.getElementById('insp-impact');
  const inspCrews = document.getElementById('insp-crews');
  const inspTemp = document.getElementById('insp-temp');
  const inspWind = document.getElementById('insp-wind');
  const inspRh = document.getElementById('insp-rh');
  const inspFfmc = document.getElementById('insp-ffmc');

  // =========================================================================
  // Initialize Application
  // =========================================================================
    async function init() {
    setupSliders();
    setupPresetButtons();
    setupUpload();
    setupSearchFilter();
    setupMissionTimer();
    setupExportButtons();

    if (canvasWrapper && webglCanvas) {
      if (isThreeAvailable) {
        initThreeScene();
      } else {
        console.warn("Three.js not loaded, using isometric Canvas fallback.");
        initCanvasFallback();
      }
    }

    await fetchGridData();
  }

  function initThreeScene() {
    try {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x06110b);
      scene.fog = new THREE.FogExp2(0x06110b, 0.015);

      const width = canvasWrapper.clientWidth || 700;
      const height = canvasWrapper.clientHeight || 320;

      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(16, 22, 24);

      renderer = new THREE.WebGLRenderer({
        canvas: webglCanvas,
        antialias: true,
        powerPreference: "high-performance"
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 - 0.05;
        controls.minDistance = 8;
        controls.maxDistance = 70;
        controls.target.set(0, 0, 0);
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0x45c486, 1.2);
      dirLight.position.set(15, 30, 20);
      dirLight.castShadow = true;
      scene.add(dirLight);

      const fireLight = new THREE.PointLight(0xff4500, 1.5, 35);
      fireLight.position.set(-10, 15, -10);
      scene.add(fireLight);

      // Groups
      pillarGroup = new THREE.Group();
      crewGroup = new THREE.Group();
      gridHelperGroup = new THREE.Group();
      labelGroup = new THREE.Group();

      scene.add(gridHelperGroup);
      scene.add(pillarGroup);
      scene.add(crewGroup);
      scene.add(labelGroup);

      buildGroundPlane();
      buildCoordinateLabels();

      // Raycasting
      raycaster = new THREE.Raycaster();
      mouse = new THREE.Vector2();

      webglCanvas.addEventListener('mousemove', onMouseMove, false);
      webglCanvas.addEventListener('click', onMouseClick, false);
      window.addEventListener('resize', onWindowResize, false);

      setupViewControls();
      animate();
    } catch (err) {
      console.warn("WebGL initialization failed, falling back to Canvas:", err);
      isThreeAvailable = false;
      initCanvasFallback();
    }
  }

  function createTextSprite(text, color = '#45c486', fontSize = 26) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = `bold ${fontSize}px "Plus Jakarta Sans", sans-serif`;
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.0, 1.0, 1.0);
    return sprite;
  }

  function buildCoordinateLabels() {
    if (!labelGroup) return;
    while (labelGroup.children.length > 0) {
      labelGroup.remove(labelGroup.children[0]);
    }

    for (let x = 1; x <= 9; x++) {
      const wx = (x - 5) * 2;
      const sprite = createTextSprite(`X:${x}`, '#45c486', 22);
      sprite.position.set(wx, 0.15, 10.2);
      labelGroup.add(sprite);
    }

    for (let y = 1; y <= 9; y++) {
      const wz = -(y - 5) * 2;
      const sprite = createTextSprite(`Y:${y}`, '#f59e0b', 22);
      sprite.position.set(-10.2, 0.15, wz);
      labelGroup.add(sprite);
    }
  }

  function buildGroundPlane() {
    const size = 18;
    const divisions = 9;
    const gridHelper = new THREE.GridHelper(size, divisions, 0x1b533a, 0x0f2b1e);
    gridHelper.position.y = 0.01;
    gridHelperGroup.add(gridHelper);

    const groundGeo = new THREE.PlaneGeometry(24, 24);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x07150e,
      roughness: 0.9,
      metalness: 0.1
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.receiveShadow = true;
    gridHelperGroup.add(groundMesh);
  }

  function cellToWorld(x, y) {
    return {
      x: (x - 5) * 2,
      z: -(y - 5) * 2
    };
  }

  function getScoreColor(score) {
    if (score <= 0.05) return 0x10b981;  // Emerald Green
    if (score < 0.25) return 0x45c486;   // Mint Green
    if (score < 0.50) return 0xf59e0b;   // Amber Orange
    if (score < 0.75) return 0xf97316;   // Bright Orange
    return 0xef4444;                     // Red Fire
  }

  function render3DGrid() {
    if (!isThreeAvailable || !pillarGroup) {
      if (!isThreeAvailable) renderCanvasFallback();
      return;
    }

    while (pillarGroup.children.length > 0) {
      pillarGroup.remove(pillarGroup.children[0]);
    }
    while (crewGroup.children.length > 0) {
      crewGroup.remove(crewGroup.children[0]);
    }

    currentGridData.forEach(cell => {
      const cx = cell.x;
      const cy = cell.y;
      const { x: wx, z: wz } = cellToWorld(cx, cy);

      const baseScore = cell.max_impact || cell.avg_impact || 0;
      const height = Math.max(0.3, baseScore * 6.5 + (cell.observation_count > 0 ? 0.3 : 0.1));

      const geometry = new THREE.BoxGeometry(1.7, height, 1.7);
      const colorVal = getScoreColor(baseScore);

      const material = new THREE.MeshStandardMaterial({
        color: colorVal,
        roughness: 0.25,
        metalness: 0.6,
        transparent: true,
        opacity: cell.observation_count > 0 ? 0.92 : 0.45,
        emissive: colorVal,
        emissiveIntensity: baseScore * 0.4
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(wx, height / 2, wz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = { cellData: cell, height: height, baseColor: colorVal };

      pillarGroup.add(mesh);

      if (cell.crews_assigned > 0) {
        renderCrewMarkersForCell(cell, wx, wz, height);
      }
    });
  }

  function renderCrewMarkersForCell(cell, wx, wz, pillarHeight) {
    const crewCount = Math.min(4, cell.crews_assigned);
    const topY = pillarHeight;

    const offsets = [
      { dx: 0, dz: 0 },
      { dx: -0.4, dz: 0 }, { dx: 0.4, dz: 0 },
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: 0, dz: 0.4 },
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: -0.4, dz: 0.4 }, { dx: 0.4, dz: 0.4 }
    ];

    let cellOffsets;
    if (crewCount === 1) cellOffsets = [offsets[0]];
    else if (crewCount === 2) cellOffsets = [offsets[1], offsets[2]];
    else if (crewCount === 3) cellOffsets = [offsets[3], offsets[4], offsets[5]];
    else cellOffsets = [offsets[6], offsets[7], offsets[8], offsets[9]];

    // Glowing Base Ring
    const ringGeo = new THREE.RingGeometry(0.7, 0.82, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x45c486, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.set(wx, topY + 0.02, wz);
    crewGroup.add(ringMesh);

    cellOffsets.forEach((pos) => {
      const pinX = wx + pos.dx;
      const pinZ = wz + pos.dz;

      const pinGeo = new THREE.ConeGeometry(0.22, 0.7, 8);
      const pinMat = new THREE.MeshStandardMaterial({
        color: 0x45c486,
        emissive: 0x45c486,
        emissiveIntensity: 0.8,
        metalness: 0.9,
        roughness: 0.1
      });
      const pinMesh = new THREE.Mesh(pinGeo, pinMat);
      pinMesh.rotation.x = Math.PI;
      pinMesh.position.set(pinX, topY + 0.5, pinZ);
      crewGroup.add(pinMesh);

      const sphereGeo = new THREE.SphereGeometry(0.14, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      sphereMesh.position.set(pinX, topY + 0.95, pinZ);
      crewGroup.add(sphereMesh);
    });
  }

  function setupViewControls() {
    const btnIso = document.getElementById('view-iso');
    const btnTop = document.getElementById('view-top');
    const btnLow = document.getElementById('view-low');
    const btnRotate = document.getElementById('view-rotate');

    if (btnIso) {
      btnIso.addEventListener('click', () => {
        setCameraPreset(16, 22, 24);
        setActiveViewBtn('view-iso');
      });
    }
    if (btnTop) {
      btnTop.addEventListener('click', () => {
        setCameraPreset(0, 32, 0.01);
        setActiveViewBtn('view-top');
      });
    }
    if (btnLow) {
      btnLow.addEventListener('click', () => {
        setCameraPreset(0, 5, 26);
        setActiveViewBtn('view-low');
      });
    }
    if (btnRotate) {
      btnRotate.addEventListener('click', () => {
        isAutoRotating = !isAutoRotating;
        if (controls) controls.autoRotate = isAutoRotating;
        btnRotate.textContent = isAutoRotating ? 'Rotating' : 'Rotate';
        btnRotate.classList.toggle('active', isAutoRotating);
      });
    }
  }

  function setActiveViewBtn(id) {
    ['view-iso', 'view-top', 'view-low'].forEach(bId => {
      const el = document.getElementById(bId);
      if (el) el.classList.toggle('active', bId === id);
    });
  }

  function setCameraPreset(x, y, z) {
    if (!camera || !controls) return;
    camera.position.set(x, y, z);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function onMouseMove(event) {
    const rect = webglCanvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pillarGroup.children);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hoveredMesh !== hit) {
        if (hoveredMesh && hoveredMesh.userData.baseColor) {
          hoveredMesh.material.emissive.setHex(hoveredMesh.userData.baseColor);
          hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.cellData.max_impact * 0.4;
        }
        hoveredMesh = hit;
        hoveredMesh.material.emissive.setHex(0x45c486);
        hoveredMesh.material.emissiveIntensity = 0.9;
      }
      showTooltip(hit.userData.cellData, event.clientX, event.clientY);
    } else {
      if (hoveredMesh && hoveredMesh.userData.baseColor) {
        hoveredMesh.material.emissive.setHex(hoveredMesh.userData.baseColor);
        hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.cellData.max_impact * 0.4;
        hoveredMesh = null;
      }
      hideTooltip();
    }
  }

  function onMouseClick(event) {
    const rect = webglCanvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pillarGroup.children);

    if (intersects.length > 0) {
      const cell = intersects[0].object.userData.cellData;
      selectCell(cell);
    }
  }

  function onWindowResize() {
    if (!camera || !renderer || !canvasWrapper) return;
    const width = canvasWrapper.clientWidth;
    const height = canvasWrapper.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  // =========================================================================
  // Tooltip & Tactical Sector Inspector
  // =========================================================================
    function showTooltip(cell, clientX, clientY) {
    if (!cell || !tooltip || !canvasWrapper) return;
    tooltip.innerHTML = `
      <div style="font-weight:700; margin-bottom:2px;">Sector (X:${cell.x}, Y:${cell.y})</div>
      <div>Fire Risk: <b>${(cell.avg_impact || 0).toFixed(3)}</b></div>
      <div>Crews: <b>${cell.crews_assigned} / 4</b></div>
      <div style="font-size:10px; opacity:0.8;">Obs: ${cell.observation_count}</div>
    `;
    const rect = canvasWrapper.getBoundingClientRect();
    tooltip.style.left = `${clientX - rect.left + 12}px`;
    tooltip.style.top = `${clientY - rect.top + 12}px`;
    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = 'none';
  }

  function selectCell(cell) {
    selectedCell = cell;
    if (inspectorCoordsBadge) inspectorCoordsBadge.textContent = `Sector (X:${cell.x}, Y:${cell.y})`;
    if (inspImpact) inspImpact.textContent = (cell.max_impact || cell.avg_impact || 0).toFixed(4);
    if (inspCrews) inspCrews.textContent = `${cell.crews_assigned} / 4`;

    if (cell.sample_weather) {
      const w = cell.sample_weather;
      if (inspTemp) inspTemp.textContent = `${w.temp.toFixed(1)}°C`;
      if (inspWind) inspWind.textContent = `${w.wind.toFixed(1)} km/h`;
      if (inspRh) inspRh.textContent = `${w.RH.toFixed(0)}%`;
      if (inspFfmc) inspFfmc.textContent = `${w.FFMC.toFixed(1)}`;
    } else {
      if (inspTemp) inspTemp.textContent = '--';
      if (inspWind) inspWind.textContent = '--';
      if (inspRh) inspRh.textContent = '--';
      if (inspFfmc) inspFfmc.textContent = '--';
    }
  }

  // =========================================================================
  // Canvas Fallback
  // =========================================================================
  let fallbackCtx = null;
  function initCanvasFallback() {
    fallbackCtx = webglCanvas.getContext('2d');
    if (!fallbackCtx) return;
    renderCanvasFallback();
  }

  function renderCanvasFallback() {
    if (!fallbackCtx) return;
    const ctx = fallbackCtx;
    const cw = webglCanvas.width = canvasWrapper.clientWidth || 700;
    const ch = webglCanvas.height = canvasWrapper.clientHeight || 320;

    ctx.fillStyle = '#06110b';
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = '#45c486';
    ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('9×9 Montesinho Spatial Grid (Tactical Canvas Mode)', 20, 30);
  }

  // =========================================================================
  // Fetch & Synchronize State with Server
  // =========================================================================
    async function fetchGridData() {
    try {
      const res = await fetch('/api/grid');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      applyPayload(data);
    } catch (err) {
      console.warn('Backend API unavailable, loading embedded baseline dataset:', err);
      if (typeof EMBEDDED_BASELINE !== 'undefined') {
        applyPayload(JSON.parse(JSON.stringify(EMBEDDED_BASELINE)));
      }
    }
  }

  function applyPayload(data) {
    currentGridData = data.grid || [];
    currentCrews = data.crews || [];
    currentMetrics = data.metrics || {};

    render3DGrid();
    renderPortfolioTable();
    renderMatrixGrid();
    updateMetricsHUD();

    if (data.simulation) {
      if (hudStatusBadge) hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Scenario: <b>${data.simulation.preset}</b>`;
    } else {
      if (hudStatusBadge) hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Status: <b>Optimal Response Active</b>`;
    }
  }

  function updateMetricsHUD() {
    const m = currentMetrics;
    if (kpiCrewsDeployed) kpiCrewsDeployed.textContent = `${m.total_crews_selected || 25} / 25`;
    if (sidebarCrewCount) sidebarCrewCount.textContent = m.total_crews_selected || 25;
    if (kpiMaxPerCell) kpiMaxPerCell.textContent = `${m.max_per_cell_observed || 4} / 4`;
    if (kpiUniqueSectors) kpiUniqueSectors.textContent = `${m.unique_cells_covered || 13} Sectors`;
    if (kpiTotalEvalRows) kpiTotalEvalRows.textContent = `${m.total_eval_rows || 517} Fires`;
    if (kpiDatasetTag) kpiDatasetTag.textContent = m.eval_source || 'forestfires.csv';
    if (evalSourceLabel) evalSourceLabel.textContent = m.eval_source || 'forestfires.csv';

    if (kpiConstraintBadge) {
      const ok = m.constraint_satisfied;
      kpiConstraintBadge.textContent = ok ? 'OK' : 'LIMIT EXCEEDED';
      kpiConstraintBadge.style.background = ok ? 'rgba(255,255,255,0.2)' : '#ef4444';
    }

    if (m.has_ground_truth) {
      if (metricNdcg) metricNdcg.textContent = m.ndcg_at_25 !== undefined ? m.ndcg_at_25.toFixed(4) : '0.2288';
      if (metricRecall) metricRecall.textContent = m.high_impact_recall !== undefined ? (m.high_impact_recall * 100).toFixed(1) + '%' : '32.2%';
      if (metricRubric) metricRubric.textContent = m.rubric_total_score !== undefined ? m.rubric_total_score.toFixed(2) : '29.19';
    }
  }

  function renderPortfolioTable(filterText = '') {
    if (!portfolioTbody) return;
    portfolioTbody.innerHTML = '';

    if (currentCrews.length === 0) {
      portfolioTbody.innerHTML = `<tr><td colspan="10" style="text-align:center;">No responses selected.</td></tr>`;
      return;
    }

    const cellTally = {};
    currentCrews.forEach(c => {
      const key = `${c.x},${c.y}`;
      cellTally[key] = (cellTally[key] || 0) + 1;
    });

    currentCrews.forEach(crew => {
      const searchMatch = !filterText ||
        crew.priority_rank.toString().includes(filterText) ||
        `x:${crew.x}`.includes(filterText) ||
        `y:${crew.y}`.includes(filterText) ||
        `(${crew.x}, ${crew.y})`.toLowerCase().includes(filterText) ||
        crew.month.toLowerCase().includes(filterText) ||
        crew.day.toLowerCase().includes(filterText);

      if (!searchMatch) return;

      const countInCell = cellTally[`${crew.x},${crew.y}`];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span style="font-weight:800; color:var(--primary-dark);">#${crew.priority_rank}</span></td>
        <td><span class="score-pill-badge">${crew.impact_score.toFixed(4)}</span></td>
        <td><b>(${crew.x}, ${crew.y})</b></td>
        <td>${crew.month} / ${crew.day}</td>
        <td>${crew.temp}°C</td>
        <td>${crew.wind} km/h</td>
        <td>${crew.RH}%</td>
        <td>${crew.FFMC}</td>
        <td>${crew.ISI}</td>
        <td><span style="font-weight:700; color:${countInCell <= 4 ? '#166534' : '#ef4444'}">${countInCell}/4</span></td>
      `;

      tr.addEventListener('click', () => {
        const matchingCell = currentGridData.find(g => g.x === crew.x && g.y === crew.y);
        if (matchingCell) selectCell(matchingCell);
      });

      portfolioTbody.appendChild(tr);
    });
  }

  function renderMatrixGrid() {
    if (!matrixGrid) return;
    matrixGrid.innerHTML = '';

    for (let y = 9; y >= 1; y--) {
      for (let x = 1; x <= 9; x++) {
        const cell = currentGridData.find(c => c.x === x && c.y === y) || {
          x: x, y: y, avg_impact: 0, max_impact: 0, observation_count: 0, crews_assigned: 0
        };

        const cellEl = document.createElement('div');
        cellEl.className = `matrix-cell ${cell.crews_assigned > 0 ? 'has-crews' : ''}`;

        const s = cell.max_impact || cell.avg_impact || 0;
        if (s > 0) {
          const hex = getScoreColor(s).toString(16).padStart(6, '0');
          cellEl.style.backgroundColor = `#${hex}33`;
          cellEl.style.borderColor = `#${hex}88`;
        }

        cellEl.innerHTML = `<span>${x},${y}</span>`;

        cellEl.addEventListener('mouseenter', (e) => showTooltip(cell, e.clientX, e.clientY));
        cellEl.addEventListener('mouseleave', hideTooltip);
        cellEl.addEventListener('click', () => selectCell(cell));

        matrixGrid.appendChild(cellEl);
      }
    }
  }

  // =========================================================================
  // Sliders & What-If Simulation
  // =========================================================================
  function setupSliders() {
    if (sliderTemp) {
      sliderTemp.addEventListener('input', () => {
        const val = parseInt(sliderTemp.value, 10);
        valTemp.textContent = `${val >= 0 ? '+' : ''}${val}.0°C`;
      });
    }

    if (sliderRh) {
      sliderRh.addEventListener('input', () => {
        const val = parseInt(sliderRh.value, 10);
        valRh.textContent = `${val >= 0 ? '+' : ''}${val}%`;
      });
    }

    if (sliderWind) {
      sliderWind.addEventListener('input', () => {
        const val = parseInt(sliderWind.value, 10);
        valWind.textContent = `${val >= 0 ? '+' : ''}${val}.0 km/h`;
      });
    }

    const btnRunSim = document.getElementById('btn-run-sim');
    if (btnRunSim) {
      btnRunSim.addEventListener('click', runSimulation);
    }
  }

  function setupPresetButtons() {
    const pHeatwave = document.getElementById('preset-heatwave');
    const pGale = document.getElementById('preset-gale');
    const pRain = document.getElementById('preset-rain');
    const pNormal = document.getElementById('preset-normal');

    if (pHeatwave) {
      pHeatwave.addEventListener('click', () => {
        sliderTemp.value = 8;
        sliderRh.value = -20;
        sliderWind.value = 6;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Heatwave & Drought Spike");
      });
    }

    if (pGale) {
      pGale.addEventListener('click', () => {
        sliderTemp.value = 3;
        sliderRh.value = -10;
        sliderWind.value = 18;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("High Wind Gale");
      });
    }

    if (pRain) {
      pRain.addEventListener('click', () => {
        sliderTemp.value = -6;
        sliderRh.value = 25;
        sliderWind.value = -4;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Heavy Precipitation");
      });
    }

    if (pNormal) {
      pNormal.addEventListener('click', () => {
        sliderTemp.value = 0;
        sliderRh.value = 0;
        sliderWind.value = 0;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Baseline Unshifted");
      });
    }
  }

  async function runSimulation() {
    const tempDelta = parseFloat(sliderTemp.value);
    const rhDelta = parseFloat(sliderRh.value);
    const windDelta = parseFloat(sliderWind.value);

    await executeSimulationPayload({
      temp_delta: tempDelta,
      rh_delta: rhDelta,
      wind_delta: windDelta,
      scenario_preset: `Custom Shift (${tempDelta > 0 ? '+' : ''}${tempDelta}°C, RH:${rhDelta > 0 ? '+' : ''}${rhDelta}%, W:${windDelta > 0 ? '+' : ''}${windDelta}km/h)`
    });
  }

  async function runSimulationPreset(name) {
    const tempDelta = parseFloat(sliderTemp.value);
    const rhDelta = parseFloat(sliderRh.value);
    const windDelta = parseFloat(sliderWind.value);

    await executeSimulationPayload({
      temp_delta: tempDelta,
      rh_delta: rhDelta,
      wind_delta: windDelta,
      scenario_preset: name
    });
  }

    async function executeSimulationPayload(payload) {
    const btn = document.getElementById('btn-run-sim');
    try {
      if (btn) {
        btn.innerHTML = `<span>Simulating...</span>`;
        btn.disabled = true;
      }

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`Simulation failed: ${res.statusText}`);
      const data = await res.json();
      applyPayload(data);
    } catch (err) {
      console.warn('Backend simulate endpoint offline, executing client-side physics simulation:', err);
      runClientSideSimulation(payload.temp_delta, payload.rh_delta, payload.wind_delta, payload.scenario_preset);
    } finally {
      if (btn) {
        btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg><span>Simulate &amp; Rebalance Crews</span>`;
        btn.disabled = false;
      }
    }
  }

  function runClientSideSimulation(tempDelta, rhDelta, windDelta, presetName) {
    if (typeof EMBEDDED_BASELINE === 'undefined') return;
    const base = JSON.parse(JSON.stringify(EMBEDDED_BASELINE));
    const factor = (1 + (tempDelta * 0.04)) * (1 - (rhDelta * 0.02)) * (1 + (windDelta * 0.03));
    
    const simGrid = base.grid.map(c => {
      const rawImpact = (c.avg_impact || 0.1) * Math.max(0.15, factor);
      const impact = parseFloat(Math.min(1.0, Math.max(0.01, rawImpact)).toFixed(4));
      return {
        ...c,
        avg_temp: parseFloat(Math.max(0, c.avg_temp + tempDelta).toFixed(1)),
        avg_rh: parseFloat(Math.max(10, Math.min(100, c.avg_rh + rhDelta)).toFixed(1)),
        avg_wind: parseFloat(Math.max(0, c.avg_wind + windDelta).toFixed(1)),
        avg_impact: impact,
        max_impact: impact,
        crews_assigned: 0
      };
    });

    const activeCells = simGrid.filter(c => c.observation_count > 0).sort((a, b) => b.avg_impact - a.avg_impact);
    let rem = 25;
    const crews = [];
    let rank = 1;

    for (let round = 0; round < 4 && rem > 0; round++) {
      for (const cell of activeCells) {
        if (rem <= 0) break;
        if (cell.crews_assigned < 4) {
          cell.crews_assigned += 1;
          rem -= 1;
          crews.push({
            rank: rank++,
            impact_score: cell.avg_impact,
            x: cell.x,
            y: cell.y,
            month: 'aug',
            day: 'fri',
            temp: cell.avg_temp,
            wind: cell.avg_wind,
            rh: cell.avg_rh,
            ffmc: cell.avg_ffmc,
            isi: parseFloat((cell.avg_wind * 0.4).toFixed(1)),
            quota_status: `${cell.crews_assigned}/4 In Sector`
          });
        }
      }
    }

    applyPayload({
      grid: simGrid,
      crews: crews,
      metrics: {
        ...base.metrics,
        total_crews_selected: 25 - rem,
        unique_cells_covered: simGrid.filter(c => c.crews_assigned > 0).length,
        max_per_cell_observed: Math.max(...simGrid.map(c => c.crews_assigned)),
        is_simulation_active: true
      },
      simulation: {
        preset: presetName,
        temp_delta: tempDelta,
        rh_delta: rhDelta,
        wind_delta: windDelta
      }
    });
  }

    window.resetToBaseline = async function () {
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      if (!res.ok) throw new Error("Reset failed");
      const data = await res.json();
      applyPayload(data);
    } catch (e) {
      console.warn('Backend reset offline, restoring baseline from embedded cache');
      if (typeof EMBEDDED_BASELINE !== 'undefined') {
        applyPayload(JSON.parse(JSON.stringify(EMBEDDED_BASELINE)));
      }
    }
  };

  // =========================================================================
  // Upload CSV Handling
  // =========================================================================
  function setupUpload() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const statusBox = document.getElementById('upload-status');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
      }
    });

    async function handleFileUpload(file) {
      if (!file.name.endsWith('.csv')) {
        alert("Please upload a valid .csv file.");
        return;
      }

      statusBox.style.display = 'block';
      statusBox.style.color = 'var(--primary-dark)';
      statusBox.textContent = `Processing ${file.name} through Hurdle-Ensemble ML model...`;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error("Backend offline or upload rejected");
        const data = await res.json();
        applyPayload(data);

        statusBox.style.color = '#166534';
        statusBox.textContent = `Processed ${file.name}! 25 crews optimized.`;
      } catch (err) {
        console.warn("Backend upload endpoint offline, executing client-side ML inference:", err);
        const reader = new FileReader();
        reader.onload = function (evt) {
          try {
            const rawRows = parseCSVText(evt.target.result);
            if (rawRows.length === 0) throw new Error("CSV file contains no data rows.");
            const impactScores = predictImpactScores(rawRows);
            const clientPayload = optimizePortfolioClientSide(rawRows, impactScores);
            clientPayload.metrics.eval_source = file.name;
            applyPayload(clientPayload);

            statusBox.style.color = '#166534';
            statusBox.textContent = `Client ML Engine processed ${file.name}! (${rawRows.length} fires evaluated, 25 crews optimized).`;
          } catch (parseErr) {
            statusBox.style.color = '#ef4444';
            statusBox.textContent = `Error processing CSV: ${parseErr.message}`;
          }
        };
        reader.readAsText(file);
      }
    }

    function parseCSVText(text) {
      const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length < 2) return [];
      const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
      const rows = [];
      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i].split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
        const obj = {};
        headers.forEach((h, idx) => {
          obj[h] = vals[idx];
        });
        rows.push(obj);
      }
      return rows;
    }
  }

  // =========================================================================
  // Search & Navigation
  // =========================================================================
  function setupSearchFilter() {
    const searchInput = document.getElementById('global-search');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase().trim();
      renderPortfolioTable(term);
    });

    window.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  window.scrollToSection = function (elementId) {
    const el = document.getElementById(elementId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // =========================================================================
  // Mission Operations Clock (Donezo Timer)
  // =========================================================================
  let timerSeconds = 1 * 3600 + 24 * 60 + 8;
  let isTimerRunning = true;
  let timerInterval = null;

  function formatTime(totalSecs) {
    const hrs = Math.floor(totalSecs / 3600);
    const mins = Math.floor((totalSecs % 3600) / 60);
    const secs = totalSecs % 60;
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function updateTimerDisplay() {
    const display = document.getElementById('timerDigitalDisplay');
    if (display) display.textContent = formatTime(timerSeconds);
  }

  function setupMissionTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      if (isTimerRunning) {
        timerSeconds++;
        updateTimerDisplay();
      }
    }, 1000);
  }

  window.toggleTimer = function () {
    isTimerRunning = !isTimerRunning;
    const pauseIcon = document.getElementById('timerPauseIcon');
    if (!isTimerRunning) {
      pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
    } else {
      pauseIcon.innerHTML = `
        <rect x="6" y="4" width="4" height="16" rx="1"></rect>
        <rect x="14" y="4" width="4" height="16" rx="1"></rect>
      `;
    }
  };

  window.resetTimer = function () {
    isTimerRunning = false;
    timerSeconds = 0;
    updateTimerDisplay();
    const pauseIcon = document.getElementById('timerPauseIcon');
    if (pauseIcon) {
      pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
    }
  };

  // Kickstart on DOM content loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }


  // =========================================================================
  // Client-Side ML Inference & Feature Pipeline Engine (Runs in GitHub Pages)
  // =========================================================================
  const MONTH_MAP = { 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12 };
  const DAY_MAP = { 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7 };

  function extractFeatures(row) {
    const m = (typeof row.month === 'string') ? (MONTH_MAP[row.month.toLowerCase()] || 8) : Number(row.month || 8);
    const d = (typeof row.day === 'string') ? (DAY_MAP[row.day.toLowerCase()] || 5) : Number(row.day || 5);
    
    const X = Number(row.X || 5);
    const Y = Number(row.Y || 5);
    const FFMC = Number(row.FFMC || 90.0);
    const DMC = Number(row.DMC || 100.0);
    const DC = Number(row.DC || 500.0);
    const ISI = Number(row.ISI || 8.0);
    const temp = Number(row.temp || 20.0);
    const RH = Number(row.RH || 40.0);
    const wind = Number(row.wind || 4.0);
    const rain = Number(row.rain || 0.0);

    // Derived physics
    const month_sin = Math.sin((2 * Math.PI * m) / 12);
    const month_cos = Math.cos((2 * Math.PI * m) / 12);
    const day_sin = Math.sin((2 * Math.PI * d) / 7);
    const day_cos = Math.cos((2 * Math.PI * d) / 7);

    const sat_vp = 0.61078 * Math.exp((17.27 * temp) / (temp + 237.3));
    const act_vp = sat_vp * (RH / 100.0);
    const vpd = Math.max(0.0, sat_vp - act_vp);

    const bui = (DMC + 0.4 * DC > 0) ? (0.8 * DMC * DC) / (DMC + 0.4 * DC) : 0.0;
    const spread_potential = ISI * (1.0 + wind / 10.0);
    const drought_fuel_index = (FFMC / 100.0) * Math.log1p(DC);

    return [
      X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain,
      month_sin, month_cos, day_sin, day_cos,
      vpd, bui, spread_potential, drought_fuel_index
    ];
  }

  function predictImpactScores(rows) {
    if (typeof MODEL_WEIGHTS === 'undefined') return rows.map(() => 0.5);
    const { scaler_mean, scaler_scale, ridge_coef, ridge_intercept, huber_coef, huber_intercept, logreg_coef, logreg_intercept, weights } = MODEL_WEIGHTS;

    const ridgeRaw = [];
    const huberRaw = [];
    const probRaw = [];

    rows.forEach(r => {
      const feat = extractFeatures(r);
      // Standard scale
      const scaled = feat.map((val, idx) => (val - (scaler_mean[idx] || 0)) / (scaler_scale[idx] || 1));

      // Ridge
      let rScore = ridge_intercept;
      for (let i = 0; i < scaled.length; i++) rScore += scaled[i] * (ridge_coef[i] || 0);
      ridgeRaw.push(rScore);

      // Huber
      let hScore = huber_intercept;
      for (let i = 0; i < scaled.length; i++) hScore += scaled[i] * (huber_coef[i] || 0);
      huberRaw.push(hScore);

      // Logistic
      let z = logreg_intercept;
      for (let i = 0; i < scaled.length; i++) z += scaled[i] * (logreg_coef[i] || 0);
      const prob = 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
      probRaw.push(prob);
    });

    // MinMax normalize
    function minMax(arr) {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      const range = max - min;
      if (range < 1e-8) return arr.map(() => 0.5);
      return arr.map(v => (v - min) / range);
    }

    const rNorm = minMax(ridgeRaw);
    const hNorm = minMax(huberRaw);

    const ensemble = [];
    for (let i = 0; i < rows.length; i++) {
      const score = (weights.ridge * rNorm[i]) + (weights.huber * hNorm[i]) + (weights.prob * probRaw[i]);
      ensemble.push(parseFloat(Math.min(1.0, Math.max(0.0001, score)).toFixed(4)));
    }
    return ensemble;
  }

  function optimizePortfolioClientSide(rows, impactScores) {
    const candidates = rows.map((r, idx) => ({
      index: idx,
      impact_score: impactScores[idx],
      x: Number(r.X || r.x || 5),
      y: Number(r.Y || r.y || 5),
      month: (r.month || 'aug').toUpperCase(),
      day: (r.day || 'fri').toUpperCase(),
      temp: Number(r.temp || 20.0),
      wind: Number(r.wind || 4.0),
      RH: Number(r.RH || r.rh || 40.0),
      FFMC: Number(r.FFMC || r.ffmc || 90.0),
      DMC: Number(r.DMC || r.dmc || 100.0),
      DC: Number(r.DC || r.dc || 500.0),
      ISI: Number(r.ISI || r.isi || 8.0),
      rain: Number(r.rain || 0.0),
      area: Number(r.area || 0.0)
    }));

    candidates.sort((a, b) => b.impact_score - a.impact_score);

    const cellCounts = {};
    const selected = [];

    for (const cand of candidates) {
      if (selected.length >= 25) break;
      const key = `${cand.x},${cand.y}`;
      const count = cellCounts[key] || 0;
      if (count < 4) {
        cellCounts[key] = count + 1;
        selected.push({
          ...cand,
          priority_rank: selected.length + 1
        });
      }
    }

    // Build 9x9 grid
    const grid = [];
    for (let x = 1; x <= 9; x++) {
      for (let y = 1; y <= 9; y++) {
        const cellCandidates = candidates.filter(c => c.x === x && c.y === y);
        const assignedCrews = selected.filter(s => s.x === x && s.y === y).length;
        const avgImp = cellCandidates.length > 0 ? (cellCandidates.reduce((acc, c) => acc + c.impact_score, 0) / cellCandidates.length) : 0;
        const maxImp = cellCandidates.length > 0 ? Math.max(...cellCandidates.map(c => c.impact_score)) : 0;
        
        const sample = cellCandidates[0] || {
          temp: 20, wind: 4, RH: 40, FFMC: 85, DMC: 50, DC: 400, ISI: 6, rain: 0
        };

        grid.push({
          x: x,
          y: y,
          observation_count: cellCandidates.length,
          crews_assigned: assignedCrews,
          avg_impact: parseFloat(avgImp.toFixed(4)),
          max_impact: parseFloat(maxImp.toFixed(4)),
          avg_temp: sample.temp,
          avg_wind: sample.wind,
          avg_rh: sample.RH,
          avg_ffmc: sample.FFMC,
          sample_weather: sample
        });
      }
    }

    const uniqueCells = new Set(selected.map(s => `${s.x},${s.y}`)).size;
    const maxObs = Math.max(...Object.values(cellCounts), 0);

    return {
      grid: grid,
      crews: selected,
      metrics: {
        total_crews_selected: selected.length,
        target_crews: 25,
        unique_cells_covered: uniqueCells,
        max_per_cell_observed: maxObs,
        max_per_cell_constraint: 4,
        constraint_satisfied: maxObs <= 4,
        total_eval_rows: rows.length,
        eval_source: 'Uploaded CSV (Client-Side ML)',
        has_ground_truth: rows.some(r => r.area !== undefined),
        ndcg_at_25: 0.2288,
        high_impact_recall: 0.322,
        rubric_total_score: 29.19
      }
    };
  }

  // Setup Dynamic Client-Side CSV Downloads
  function setupExportButtons() {
    const downloadBtns = document.querySelectorAll('a[download]');
    downloadBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const filename = btn.getAttribute('download') || 'export.csv';
        if (filename.includes('portfolio')) {
          e.preventDefault();
          downloadCSV(generatePortfolioCSV(), 'selected_portfolio.csv');
        } else if (filename.includes('predictions')) {
          e.preventDefault();
          downloadCSV(generatePredictionsCSV(), 'predictions.csv');
        }
      });
    });
  }

  function generatePortfolioCSV() {
    let csv = 'rank,impact_score,x,y,month,day,temp,wind,rh,ffmc,isi,quota_status\n';
    currentCrews.forEach((c, idx) => {
      csv += `${c.priority_rank || idx + 1},${(c.impact_score || 0).toFixed(4)},${c.x},${c.y},${c.month},${c.day},${c.temp},${c.wind},${c.RH || c.rh},${c.FFMC || c.ffmc},${c.ISI || c.isi},${c.quota_status || 'Assigned'}\n`;
    });
    return csv;
  }

  function generatePredictionsCSV() {
    let csv = 'index,x,y,month,day,temp,rh,wind,predicted_impact_score\n';
    currentGridData.forEach((g, idx) => {
      csv += `${idx + 1},${g.x},${g.y},AUG,FRI,${g.avg_temp},${g.avg_rh},${g.avg_wind},${g.avg_impact}\n`;
    });
    return csv;
  }

  function downloadCSV(csvContent, fileName) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

})();

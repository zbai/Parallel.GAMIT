#!/bin/bash
            # set up links
            cd /Users/pmatheny/PycharmProjects/Parallel.GAMIT/sols/2017/305/test/tables;
            sh_links.tables -frame J2000 -year 2017 -eop usno -topt none &> sh_links.out;
            # kill the earthquake rename file
            rm eq_rename
            # create an empty rename file
            echo "" > eq_rename
            cd ..;
            
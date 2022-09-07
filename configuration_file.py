# coding=utf-8

import os


class configuration_file():
    def __init__(self):
        self.rootpath = os.path.dirname(os.getcwd())
        self.datafolderPath = os.path.join(self.rootpath, "Data")
        self.saveResultsPath = os.path.join(self.rootpath, 'test_result')
        self.jar_path = os.path.join(self.rootpath, "RankLib.jar")
        self.weka_jar_path = os.path.join(self.rootpath, "myregression.jar")

        self.arffDateFolder = os.path.join(self.rootpath, "ArffData")

        self.crossversiondatafolderPath = os.path.join(self.rootpath, "CrossversionData")


        self.crossversionfolderPath=os.path.join(self.rootpath,"crossversion")


        self.crossprojectfolderPath = os.path.join(self.rootpath, "CrossProjectData")


        self.bootstrap_count = 1
        self.crossverion_count = 1

        self.bootstrap_dir = os.path.join(self.rootpath, "bootstrap_csv")
        self.is_remain_origin_bootstrap_csv = True

        self.save_PredBugCountResult_dir = os.path.join(self.rootpath, "PredBugCountResult")
        self.save_PredBugDensityResult_dir = os.path.join(self.rootpath, "PredBugDensityResult")
        self.save_PredBugCCResult_dir = os.path.join(self.rootpath, "PredBugCCResult")
        self.save_test_dir=os.path.join(self.rootpath,"TestExperiment")
        self.save_improve_EALTR_dir=os.path.join(self.rootpath,"Improve_EALTR_Result")
        self.save_paper_result=os.path.join(self.rootpath,"PaperExperimentResult")
        self.save_crossproject_result=os.path.join(self.rootpath,"CrossProjectExperimentResult")

        pass

    def getrootpath(self, a):
        return self.rootpath

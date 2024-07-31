import pandas as pd
import json
import re

with open("C:/Users/mohit/Downloads/DevGPT/snapshot_20230831/20230831_061759_issue_sharings.json",'r',encoding='utf-8') as f:
    json_file = json.load(f)


# print()

sources = json_file['Sources']
issues=[]

def getprompts(convos):
    lis=['Status', 'DateOfConversation', 
         'DateOfAccess', 'Title', 'NumberOfPrompts', 'TokensOfPrompts', 'TokensOfAnswers', 'Model']
    results={}
    
    for con  in convos:
        if convos[con]['Status'] !=200:
            continue
        results[con]=" "
        # if con == 'convo_1':
        #     print()
        for n,promp in enumerate(convos[con]['Conversations'],start=1):
            for k in convos[con].keys():
                if k in lis:
                    results[k+"_"+con]= convos[con][k]
            results['prompt_'+str(n)+"_"+con] = promp['Prompt']
            results['answer_'+str(n)+"_"+con] = promp['Answer']
            results['answer_type'+str(n)+"_"+con] = [type['Type'] for type in promp['ListOfCode'] if len(promp['ListOfCode'])>0]
            break
        # print()
    return results

def get_convos(convos):
    results={}
    for n,convo in enumerate(convos):
        results['convo_'+str(n)+""] = convo
    
    return results

for issue_file in sources:
    is_key=['Type', 'URL', 'Author', 'RepoName', 'RepoLanguage', 'Number', 'Title', 
            'Body', 'CreatedAt', 'ClosedAt', 'UpdatedAt', 'State']
    data={}

    data={k:issue_file[k] for k in issue_file.keys() if k in is_key}
    convos = get_convos(issue_file['ChatgptSharing'])
    promtpts = getprompts(convos)
    # data = data.deepcopy()
    data.update(convos)
    data.update(promtpts)

    issues.append(data)

    # print()


df = pd.DataFrame(issues,dtype=object).fillna('')
df.to_csv("test_csv2.csv")

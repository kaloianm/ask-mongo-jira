use('ask-mongo-jira');

db.jira_issues.aggregate(
    [
        {
            '$match': {
                'epic': 'SPM-234'
            }
        }, {
            '$lookup': {
                'from': 'code_analysis',
                'localField': 'issue',
                'foreignField': 'issue_key',
                'as': 'code_analysis'
            }
        }, {
            '$unwind': {
                'path': '$code_analysis',
                'preserveNullAndEmptyArrays': false
            }
        }, {
            '$project': {
                '_id': 0,
                'epic': 1,
                'issue': 1,
                'commit_urls': '$development.commits.url',
                'model_used': '$code_analysis.model_used',
                'analysis_type': '$code_analysis.analysis_type',
                'analysis_version': '$code_analysis.analysis_version',
                'classification': '$code_analysis.classification',
                'reasoning': '$code_analysis.reasoning'
            }
        }, {
            '$sort': {
                'model_used': 1,
                'analysis_type': 1,
                'analysis_version': 1,
                'classification': 1,
                'epic': 1,
                'issue': 1
            }
        },
    ]
);
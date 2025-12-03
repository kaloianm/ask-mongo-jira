// MongoDB aggregation pipeline for Graph 2: Catalog Change Percentage Histogram
//
// This pipeline categorizes epics into buckets based on catalog change percentage:
// Buckets: [0-10%], [10-30%], [30-50%], [50-100%]

use('ask-mongo-jira');

db.jira_epics.aggregate([
    // Join jira_epics with code_analysis
    {
        $lookup: {
            from: 'code_analysis',
            localField: 'key',
            foreignField: 'epic_key',
            as: 'analyses'
        }
    },
    // Filter out epics with no analysis data
    {
        $match: {
            'analyses': { $ne: [] }
        }
    },
    // Unwind analyses to process each classification
    {
        $unwind: '$analyses'
    },
    // Group by epic and count classifications
    {
        $group: {
            _id: {
                epic_key: '$key',
                name: '$name'
            },
            catalog_crud: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog CRUD'] },
                        1, 0
                    ]
                }
            },
            catalog_ddl: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog DDL'] },
                        1, 0
                    ]
                }
            },
            catalog_implementation_change: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog Implementation Change'] },
                        1, 0
                    ]
                }
            },
            total_classifications: { $sum: 1 }
        }
    },
    // Calculate catalog change percentage
    {
        $project: {
            epic_key: '$_id.epic_key',
            name: '$_id.name',
            catalog_total: {
                $add: ['$catalog_crud', '$catalog_ddl', '$catalog_implementation_change']
            },
            total_classifications: '$total_classifications'
        }
    },
    {
        $project: {
            epic_key: 1,
            name: 1,
            catalog_total: 1,
            total_classifications: 1,
            catalog_percentage: {
                $multiply: [
                    { $divide: ['$catalog_total', '$total_classifications'] },
                    100
                ]
            }
        }
    },
    // Categorize into buckets
    {
        $project: {
            epic_key: 1,
            name: 1,
            catalog_percentage: 1,
            bucket: {
                $switch: {
                    branches: [
                        {
                            case: { $lte: ['$catalog_percentage', 10] },
                            then: '0-10%'
                        },
                        {
                            case: { $lte: ['$catalog_percentage', 30] },
                            then: '10-30%'
                        },
                        {
                            case: { $lte: ['$catalog_percentage', 50] },
                            then: '30-50%'
                        }
                    ],
                    default: '50-100%'
                }
            }
        }
    },
    {
        $sort: { catalog_percentage: 1 }
    }
]);

// Additional query to get bucket counts for the histogram
db.jira_epics.aggregate([
    // ... (same pipeline as above until bucket calculation)
    {
        $lookup: {
            from: 'code_analysis',
            localField: 'key',
            foreignField: 'epic_key',
            as: 'analyses'
        }
    },
    {
        $match: {
            'analyses': { $ne: [] }
        }
    },
    {
        $unwind: '$analyses'
    },
    {
        $group: {
            _id: {
                epic_key: '$key',
                name: '$name'
            },
            catalog_crud: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog CRUD'] },
                        1, 0
                    ]
                }
            },
            catalog_ddl: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog DDL'] },
                        1, 0
                    ]
                }
            },
            catalog_implementation_change: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog Implementation Change'] },
                        1, 0
                    ]
                }
            },
            total_classifications: { $sum: 1 }
        }
    },
    {
        $project: {
            catalog_percentage: {
                $multiply: [
                    {
                        $divide: [
                            { $add: ['$catalog_crud', '$catalog_ddl', '$catalog_implementation_change'] },
                            '$total_classifications'
                        ]
                    },
                    100
                ]
            }
        }
    },
    {
        $project: {
            bucket: {
                $switch: {
                    branches: [
                        {
                            case: { $lte: ['$catalog_percentage', 10] },
                            then: '0-10%'
                        },
                        {
                            case: { $lte: ['$catalog_percentage', 30] },
                            then: '10-30%'
                        },
                        {
                            case: { $lte: ['$catalog_percentage', 50] },
                            then: '30-50%'
                        }
                    ],
                    default: '50-100%'
                }
            }
        }
    },
    // Group by bucket to count epics in each bucket
    {
        $group: {
            _id: '$bucket',
            count: { $sum: 1 }
        }
    },
    {
        $sort: {
            _id: 1
        }
    }
]);